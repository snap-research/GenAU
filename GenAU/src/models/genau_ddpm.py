import os
import numpy as np
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
import datetime
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
from multiprocessing.sharedctypes import Value
import time

import hashlib
import soundfile as sf

import torch
import torch.distributed as dist
import torch.nn as nn

import pytorch_lightning as pl
from torchvision.utils import make_grid
from pytorch_lightning.utilities.rank_zero import rank_zero_only
# requires latest versin of torch
from torch.utils.flop_counter import FlopCounterMode

from src.modules.conditional.conditional_models import *
from src.utilities.model.model_util import (
    exists,
    default,
    count_params,
    instantiate_from_config,
)

from src.utilities.diffusion.diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)

from src.modules.diffusionmodules.ema import LitEma
from src.modules.diffusionmodules.distributions import (
    DiagonalGaussianDistribution,
)
from src.modules.latent_diffusion.ddim import DDIMSampler
from src.modules.latent_diffusion.plms import PLMSSampler
from src.modules.data_representation.data_entries import DataEntries
from src.modules.data_representation.data_entry import DataEntry
from src.tools.training_utils import disabled_train
from src.tools.download_manager import get_checkpoint_path

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        sampling_rate=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        scale_input=1.0,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        evaluator=None,
        backbone_type='unet',
        **kwargs
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        self.state = None
        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        assert sampling_rate is not None
        self.validation_folder_names = {}
        self.validation_tags = []
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.sampling_rate = sampling_rate
        self.global_config = kwargs
        self.scale_input = scale_input
        
        # only needed for clap reranking and comptuing clap score for evaluation
        clap_ckpt_path = get_checkpoint_path('music_speech_audioset_epoch_15_esc_89.98') # IMPORTANT TODO:
        self.clap = CLAPAudioEmbeddingClassifierFreev2(
            pretrained_path=clap_ckpt_path,
            sampling_rate=self.sampling_rate,
            embed_mode="audio",
            amodel="HTSAT-base",
        )

        if self.global_rank == 0:
            if isinstance(evaluator, dict):
                self.evaluator = instantiate_from_config(evaluator)
            else:
                self.evaluator = evaluator

        self.initialize_param_check_toolkit()

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size

        self.backbone_type = backbone_type
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config,
                                     conditioning_key,
                                    backbone_type=self.backbone_type,
                                    scale_by_std=self.scale_input != 1.0)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"[INFO] Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
            
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.logvar = nn.Parameter(self.logvar, requires_grad=False)

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None
        self.logger_version = None

        self.label_indices_total = None
        # To avoid the system cannot find metric value for checkpoint
        self.metrics_buffer = {
            "val/kullback_leibler_divergence_sigmoid": 15.0,
            "val/kullback_leibler_divergence_softmax": 10.0,
            "val/psnr": 0.0,
            "val/ssim": 0.0,
            "val/inception_score_mean": 1.0,
            "val/inception_score_std": 0.0,
            "val/kernel_inception_distance_mean": 0.0,
            "val/kernel_inception_distance_std": 0.0,
            "val/frechet_inception_distance": 133.0,
            "val/frechet_audio_distance": 32.0,
        }
        self.initial_learning_rate = None
        self.test_data_subset_path = None
        self.fwd_flops = None


    def get_log_dir(self):
        return os.path.join(
            self.logger_save_dir, self.logger_exp_group_name, self.logger_exp_name
        )

    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_group_name = exp_group_name
        self.logger_exp_name = exp_name

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        # self.register_buffer(
        #     "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        # )
        # self.register_buffer(
        #     "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        # )
        # self.register_buffer(
        #     "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        # )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_variance = (1 - self.v_posterior) * betas * (
        #     1.0 - alphas_cumprod_prev
        # ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        # self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # self.register_buffer(
        #     "posterior_log_variance_clipped",
        #     to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        # )
        # self.register_buffer(
        #     "posterior_mean_coef1",
        #     to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        # )
        # self.register_buffer(
        #     "posterior_mean_coef2",
        #     to_torch(
        #         (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        #     ),
        # )

        # if self.parameterization == "eps":
        #     lvlb_weights = self.betas**2 / (
        #         2
        #         * self.posterior_variance
        #         * to_torch(alphas)
        #         * (1 - self.alphas_cumprod)
        #     )
        # elif self.parameterization == "x0":
        #     lvlb_weights = (
        #         0.5
        #         * np.sqrt(torch.Tensor(alphas_cumprod))
        #         / (2.0 * 1 - torch.Tensor(alphas_cumprod))
        #     )
        # elif self.parameterization == "v":
        #     lvlb_weights = torch.ones_like(
        #         self.betas**2
        #         / (
        #             2
        #             * self.posterior_variance
        #             * to_torch(alphas)
        #             * (1 - self.alphas_cumprod)
        #         )
        #     )
        # else:
        #     raise NotImplementedError("mu not supported")
        # lvlb_weights[0] = lvlb_weights[1]
        # self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        # assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None, use_ema=True):
        if use_ema:
            print("[INFO] loading EMA")
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        print(
            f"[INFO] Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"[WARNING] Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"[WARNING] Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        shape = (batch_size, channels, self.latent_t_size, self.latent_f_size)
        channels = self.channels
        return self.p_sample_loop(shape, return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start * self.scale_input
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        if self.original_elbo_weight != 0:
            loss_vlb = (self.lvlb_weights[t] * loss).mean()
            loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})
            loss = loss_simple + self.original_elbo_weight * loss_vlb

            loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        fname, text, waveform, stft, fbank = (
            batch.get("fname", None),
            batch.get("text", None),
            batch.get("waveform", None),
            batch.get("stft", None),
            batch.get("log_mel_spec", None),
        )
        ret = {}

        ret["fbank"] = (
            fbank.unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        ) if fbank is not None and not isinstance(fbank, list) else []
        ret["stft"] = stft.to(memory_format=torch.contiguous_format).float() if stft is not None and not isinstance(stft, list) else []
        ret["waveform"] = waveform.to(memory_format=torch.contiguous_format).float() if waveform is not None and not isinstance(waveform, list) else []
        ret["text"] = list(text) if text is not None else []
        ret["fname"] = fname if fname is not None else []

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def warmup_step(self):
        if self.initial_learning_rate is None:
            self.initial_learning_rate = self.learning_rate

        # Only the first parameter group
        if self.global_step <= self.warmup_steps:
            if self.global_step == 0:
                print(
                    "Warming up learning rate start with %s"
                    % self.initial_learning_rate
                )
            self.trainer.optimizers[0].param_groups[0]["lr"] = (
                self.global_step / self.warmup_steps
            ) * self.initial_learning_rate
        else:
            self.trainer.optimizers[0].param_groups[0][
                "lr"
            ] = self.initial_learning_rate

    
    def measure_flops(self, sample_batch):
        # only considers forward flops
        # TODO: requires last version of pytorch
        batch_size = sample_batch["log_mel_spec"].shape[0]
        with torch.no_grad(): 
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                self.shared_step(sample_batch)
            self.fwd_flops = flop_counter.get_total_flops() / 1e9 / batch_size
            print(f"[INFO] the model consumes {self.fwd_flops} FLOPs per sample in a single forward call")

        return self.fwd_flops


    def training_step(self, batch, batch_idx):
        self.random_clap_condition()
        if self.optimizer_config is None:
            self.warmup_step()

        if len(self.metrics_buffer.keys()) > 0:
            for k in self.metrics_buffer.keys():
                self.log(
                    k,
                    self.metrics_buffer[k],
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )
            self.metrics_buffer = {}

        loss, loss_dict = self.shared_step(batch)
        self.batch_size = batch["log_mel_spec"].shape[0]
        loss_dict['global_step'] = float(self.global_step)
        if self.fwd_flops is not None:
            loss_dict['gflops'] = float(self.fwd_flops * self.global_step * self.batch_size)

        loss_dict['lr_abs'] = float(self.trainer.optimizers[0].param_groups[0]["lr"])

        self.log_dict(
            {k: float(v) for k, v in loss_dict.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def random_clap_condition(self, text_prop=0.5):
        # This function is only used during training, let the CLAP model to use both text and audio as condition
        assert self.training == True

        for key in self.cond_stage_model_metadata.keys():
            metadata = self.cond_stage_model_metadata[key] # config for the first condition (normally only CLAP is used as a condition)
            model_idx, cond_stage_key, conditioning_key = (
                metadata["model_idx"],
                metadata["cond_stage_key"],
                metadata["conditioning_key"],
            )

            # If we use CLAP as condition, we might use audio for training, but we also must use text for evaluation
            if isinstance(
                self.cond_stage_models[model_idx], CLAPAudioEmbeddingClassifierFreev2
            ):
                self.cond_stage_model_metadata[key][
                    "cond_stage_key_orig"
                ] = self.cond_stage_model_metadata[key]["cond_stage_key"] # save what was the original conditioning key
                self.cond_stage_model_metadata[key][
                    "embed_mode_orig"
                ] = self.cond_stage_models[model_idx].embed_mode # original emb mode
                if torch.randn(1).item() < text_prop:
                    self.cond_stage_model_metadata[key]["cond_stage_key"] = "text"
                    self.cond_stage_models[model_idx].embed_mode = "text"
                else:
                    self.cond_stage_model_metadata[key]["cond_stage_key"] = "waveform"
                    self.cond_stage_models[model_idx].embed_mode = "audio"

    def on_validation_epoch_start(self) -> None:
        # Use text as condition during validation
        for key in self.cond_stage_model_metadata.keys():
            metadata = self.cond_stage_model_metadata[key]
            model_idx, cond_stage_key, conditioning_key = (
                metadata["model_idx"],
                metadata["cond_stage_key"],
                metadata["conditioning_key"],
            )

            # If we use CLAP as condition, we might use audio for training, but we also must use text for evaluation
            if isinstance(
                self.cond_stage_models[model_idx], CLAPAudioEmbeddingClassifierFreev2
            ):
                self.cond_stage_model_metadata[key][
                    "cond_stage_key_orig"
                ] = self.cond_stage_model_metadata[key]["cond_stage_key"]
                self.cond_stage_model_metadata[key][
                    "embed_mode_orig"
                ] = self.cond_stage_models[model_idx].embed_mode
                print(
                    "Change the model original cond_keyand embed_mode %s, %s to text during evaluation"
                    % (
                        self.cond_stage_model_metadata[key]["cond_stage_key_orig"],
                        self.cond_stage_model_metadata[key]["embed_mode_orig"],
                    )
                )
                self.cond_stage_model_metadata[key]["cond_stage_key"] = "text"
                self.cond_stage_models[model_idx].embed_mode = "text"

            if isinstance(
                self.cond_stage_models[model_idx], CLAPGenAudioMAECond
            ) or isinstance(self.cond_stage_models[model_idx], SequenceGenAudioMAECond):
                self.cond_stage_model_metadata[key][
                    "use_gt_mae_output_orig"
                ] = self.cond_stage_models[model_idx].use_gt_mae_output
                self.cond_stage_model_metadata[key][
                    "use_gt_mae_prob_orig"
                ] = self.cond_stage_models[model_idx].use_gt_mae_prob
                print("Change the model condition to the predicted AudioMAE tokens")
                self.cond_stage_models[model_idx].use_gt_mae_output = False
                self.cond_stage_models[model_idx].use_gt_mae_prob = 0.0
        
        
        self.validation_folder_names = {"val": self.get_validation_folder_name(
                                            self.evaluation_params["unconditional_guidance_scale"], 
                                            self.evaluation_params["ddim_sampling_steps"],
                                            self.evaluation_params["n_candidates_per_samples"]),
                                        "wo_ema": self.get_validation_folder_name(
                                            self.evaluation_params["unconditional_guidance_scale"], 
                                            self.evaluation_params["ddim_sampling_steps"],
                                            self.evaluation_params["n_candidates_per_samples"],
                                            tag="_wo_ema"),
                                        "uncond": self.get_validation_folder_name(
                                            1.0, 
                                            self.evaluation_params["ddim_sampling_steps"],
                                            self.evaluation_params["n_candidates_per_samples"],
                                            tag="_uncond"),}
        dist.barrier()

        self.val_start_time = time.time()
        return super().on_validation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.fwd_flops is None: 
            self.fwd_flops = self.measure_flops(batch)

        cfg_scale = self.evaluation_params["unconditional_guidance_scale"]
        ddim_steps = self.evaluation_params["ddim_sampling_steps"]
        n_cands = self.evaluation_params["n_candidates_per_samples"]
        exp_name = self.validation_folder_names["val"]
        
        print("[INFO] Logging with the EMA model")
        waveform_save_path, all_clap_sim, best_clap_sim = self.generate_sample(
            [batch],
            name=exp_name,
            unconditional_guidance_scale=cfg_scale,
            ddim_steps=ddim_steps,
            n_gen=n_cands,
            use_ema=self.use_ema
        )

        loss_dict = {}
        if all_clap_sim is not None:
            loss_dict['val/clap_sim_mean'] = torch.mean(all_clap_sim).item()
        
        if best_clap_sim is not None:
            loss_dict['val/clap_sim_mean_w_clap_reranking'] = torch.mean(best_clap_sim).item()
        
        self.log_dict(
                {f"{k}": float(v) for k, v in loss_dict.items()},
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )
        
        # log loss at different timestamp ranges to understand model performance on different levels
        seg_len = self.num_timesteps // self.num_val_sampled_timestamps
        total_loss_simple = 0 # loss_simple
        for i in range(self.num_val_sampled_timestamps):
            t_range = [i * seg_len, (i+1) * seg_len]
            loss, loss_dict = self.shared_step(batch, t_range=t_range)
            loss_dict["global_step"] = float(self.global_step)

            if self.fwd_flops is not None:
                loss_dict["gflops"] = float(self.fwd_flops * self.global_step * self.batch_size)
            
            total_loss_simple += loss_dict['val/loss_simple']
            self.log_dict(
                {f"{k}_t_{t_range[0]}_{t_range[1]}": float(v) for k, v in loss_dict.items()},
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )
        self.log("val/total_loss_simple", total_loss_simple/self.num_val_sampled_timestamps, on_step=False, on_epoch=True)
        
        ######################## samples without the EMA model #####################
        if self.validate_wo_ema:
            print("[INFO] Logging without the EMA model")
            exp_name = self.validation_folder_names["wo_ema"]

            waveform_save_path, all_clap_sim, best_clap_sim = self.generate_sample(
                [batch],
                name=exp_name,
                unconditional_guidance_scale=cfg_scale,
                ddim_steps=ddim_steps,
                n_gen=n_cands,
                use_ema=False
            )

            loss_dict = {}
            if all_clap_sim is not None:
                loss_dict['val/clap_sim_mean'] = torch.mean(all_clap_sim).item()
            
            if best_clap_sim is not None:
                loss_dict['val/clap_sim_mean_w_clap_reranking'] = torch.mean(best_clap_sim).item()

            self.log_dict(
                    {f"wo_ema/{k}": float(v) for k, v in loss_dict.items()},
                    prog_bar=True,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
            
            # log loss at different ranges
            seg_len = self.num_timesteps // self.num_val_sampled_timestamps
            total_loss_simple = 0
            for i in range(self.num_val_sampled_timestamps):
                t_range = [i * seg_len, (i+1) * seg_len]
                loss, loss_dict = self.shared_step(batch, t_range=t_range)
                loss_dict["global_step"] = float(self.global_step)        
                total_loss_simple += loss_dict['val/loss_simple']
                self.log_dict(
                    {f"wo_ema/{k}_t_{t_range[0]}_{t_range[1]}": float(v) for k, v in loss_dict.items()},
                    prog_bar=True,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )

            self.log("wo_ema/val/total_loss_simple", total_loss_simple/self.num_val_sampled_timestamps, on_step=False, on_epoch=True)

        ######################## samples with the uncondiational model #####################
        if self.validate_uncond:
            print("[INFO] Logging unconditional model")
            cfg_scale = 1.0
            ddim_steps = self.evaluation_params["ddim_sampling_steps"]
            n_cands = self.evaluation_params["n_candidates_per_samples"]
            exp_name = self.validation_folder_names['uncond']

            waveform_save_path, all_clap_sim, best_clap_sim= self.generate_sample(
                [batch],
                name=exp_name,
                unconditional_guidance_scale=cfg_scale,
                ddim_steps=ddim_steps,
                n_gen=n_cands,
                use_ema=self.use_ema
            )

            loss_dict = {}
            if all_clap_sim is not None:
                loss_dict['val/clap_sim_mean'] = torch.mean(all_clap_sim).item()
            
            if best_clap_sim is not None:
                loss_dict['val/clap_sim_mean_w_clap_reranking'] = torch.mean(best_clap_sim).item()
            
            self.log_dict(
                    {f"unconditional/{k}": float(v) for k, v in loss_dict.items()},
                    prog_bar=True,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
            # log loss at different ranges
            total_loss_simple = 0
            seg_len = self.num_timesteps // self.num_val_sampled_timestamps
            for i in range(self.num_val_sampled_timestamps):
                t_range = [i * seg_len, (i+1) * seg_len]
                loss, loss_dict = self.shared_step(batch, t_range=t_range)
                loss_dict["global_step"] = float(self.global_step)        
                total_loss_simple += loss_dict['val/loss_simple']
                
                self.log_dict(
                    {f"unconditional/{k}_t_{t_range[0]}_{t_range[1]}": float(v) for k, v in loss_dict.items()},
                    prog_bar=True,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
            self.log("unconditional/val/total_loss_simple", total_loss_simple/self.num_val_sampled_timestamps, on_step=False, on_epoch=True)
            

    def get_validation_folder_name(self, unconditional_guidance_scale=None, ddim_sampling_steps=None, n_candidates_per_samples=None, step=None, tag=""):
        now = datetime.datetime.now()
        timestamp = now.strftime("%m-%d-%H:%M")
        if step is None:
            step = self.global_step
        return "val_%s_%s_cfg_scale_%s_ddim_%s_n_cand_%s%s" % (
            step,
            timestamp,
            unconditional_guidance_scale or self.evaluation_params["unconditional_guidance_scale"],
            ddim_sampling_steps or self.evaluation_params["ddim_sampling_steps"],
            n_candidates_per_samples or self.evaluation_params["n_candidates_per_samples"],
            tag
        )

    def on_validation_epoch_end(self) -> None:
        print("self.validation_folder_names", self.validation_folder_names, "rank", self.global_rank)
        if self.global_rank == 0 and self.evaluator is not None:
                assert (
                    self.test_data_subset_path is not None
                ), "Please set test_data_subset_path before validation so that model have a target folder"
                try:
                    for tag, name in self.validation_folder_names.items():
                        print("Performaing evaluation for experiment", name)
                        waveform_save_path = os.path.join(self.get_log_dir(), name)
                        if (
                            os.path.exists(waveform_save_path)
                            and len(os.listdir(waveform_save_path)) > 0
                        ):

                            metrics = self.evaluator.main(
                                waveform_save_path,
                                self.test_data_subset_path,
                            )
                            
                            self.metrics_buffer = {
                                (f"{tag}/val" + k): float(v) for k, v in metrics.items()
                            }
                            
                        else:
                            print(
                                "The target folder for evaluation does not exist: %s"
                                % waveform_save_path
                            )
                except Exception as e:
                    print("[ERROR] An error encountered during evaluation: ", e)

        # Very important or the program may fail
        torch.cuda.synchronize()

        for key in self.cond_stage_model_metadata.keys():
            metadata = self.cond_stage_model_metadata[key]
            model_idx, cond_stage_key, conditioning_key = (
                metadata["model_idx"],
                metadata["cond_stage_key"],
                metadata["conditioning_key"],
            )

            if isinstance(
                self.cond_stage_models[model_idx], CLAPAudioEmbeddingClassifierFreev2
            ):
                self.cond_stage_model_metadata[key][
                    "cond_stage_key"
                ] = self.cond_stage_model_metadata[key]["cond_stage_key_orig"]
                self.cond_stage_models[
                    model_idx
                ].embed_mode = self.cond_stage_model_metadata[key]["embed_mode_orig"]
                print(
                    "Change back the embedding mode to %s %s"
                    % (
                        self.cond_stage_model_metadata[key]["cond_stage_key"],
                        self.cond_stage_models[model_idx].embed_mode,
                    )
                )

            if isinstance(
                self.cond_stage_models[model_idx], CLAPGenAudioMAECond
            ) or isinstance(self.cond_stage_models[model_idx], SequenceGenAudioMAECond):
                self.cond_stage_models[
                    model_idx
                ].use_gt_mae_output = self.cond_stage_model_metadata[key][
                    "use_gt_mae_output_orig"
                ]
                self.cond_stage_models[
                    model_idx
                ].use_gt_mae_prob = self.cond_stage_model_metadata[key][
                    "use_gt_mae_prob_orig"
                ]
                print(
                    "Change the AudioMAE condition setting to %s (Use gt) %s (gt prob)"
                    % (
                        self.cond_stage_models[model_idx].use_gt_mae_output,
                        self.cond_stage_models[model_idx].use_gt_mae_prob,
                    )
                )

        self.log("time/val_epoch", time.time() - self.val_start_time, on_step=False, on_epoch=True, logger=True)
        return super().on_validation_epoch_end()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        print("Log directory: ", self.get_log_dir())
        self.train_start_time = time.time()
    
    def on_train_epoch_end(self):
        self.log("time/train_epoch", time.time() - self.train_start_time, on_step=False, on_epoch=True, logger=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting", use_ema=self.use_ema):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def initialize_param_check_toolkit(self):
        self.tracked_steps = 0
        self.param_dict = {}

    def statistic_require_grad_tensor_number(self, module, name=None):
        requires_grad_num = 0
        total_num = 0
        require_grad_tensor = None
        for p in module.parameters():
            if p.requires_grad:
                requires_grad_num += 1
                if require_grad_tensor is None:
                    require_grad_tensor = p
            total_num += 1
        print(
            "Module: [%s] have %s trainable parameters out of %s total parameters (%.2f)"
            % (name, requires_grad_num, total_num, requires_grad_num / total_num)
        )
        return require_grad_tensor

    def check_module_param_update(self):
        if self.tracked_steps == 0:
            for name, module in self.named_children():
                try:
                    require_grad_tensor = self.statistic_require_grad_tensor_number(
                        module, name=name
                    )
                    if require_grad_tensor is not None:
                        self.param_dict[name] = require_grad_tensor.clone()
                    else:
                        print("==> %s does not requires grad" % name)
                except Exception as e:
                    print("%s does not have trainable parameters: %s" % (name, e))
                    continue

        if self.tracked_steps % 5000 == 0:
            for name, module in self.named_children():
                try:
                    require_grad_tensor = self.statistic_require_grad_tensor_number(
                        module, name=name
                    )

                    if require_grad_tensor is not None:
                        print(
                            "===> Param diff %s: %s; Size: %s"
                            % (
                                name,
                                torch.sum(
                                    torch.abs(
                                        self.param_dict[name] - require_grad_tensor
                                    )
                                ),
                                require_grad_tensor.size(),
                            )
                        )
                    else:
                        print("%s does not requires grad" % name)
                except Exception as e:
                    print("%s does not have trainable parameters: %s" % (name, e))
                    continue

        self.tracked_steps += 1


class GenAu(DDPM):
    """main class"""
    def __init__(
        self,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        noise_scheduler_config=None,
        cond_stage_key="image",
        optimize_ddpm_parameter=True,
        unconditional_prob_cfg=0.1,
        warmup_steps=10000,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        evaluation_params={},
        scale_by_std=False,
        base_learning_rate=None,
        optimizer_config=None,
        validate_uncond=False,
        validate_wo_ema=False,
        num_val_sampled_timestamps=1,
        dataset2id=None,
        dataset_embed_dim=None,
        log_melspectrogran=False,
        *args,
        **kwargs,
    ):
        self.learning_rate = base_learning_rate
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.warmup_steps = warmup_steps
        self.optimizer_config = optimizer_config
        self.noise_scheduler_config = noise_scheduler_config
        self.batch_size = 0
        self.validate_uncond = validate_uncond
        self.validate_wo_ema = validate_wo_ema
        self.num_val_sampled_timestamps=num_val_sampled_timestamps
        self.dataset_embed_dim = dataset_embed_dim
        self.dataset2id = dataset2id
        self.evaluation_params = evaluation_params
        self.log_melspectrogran = log_melspectrogran

        if optimize_ddpm_parameter:
            if unconditional_prob_cfg == 0.0:
                "You choose to optimize DDPM. The classifier free guidance scale should be 0.1"
                unconditional_prob_cfg = 0.1
        else:
            if unconditional_prob_cfg == 0.1:
                "You choose not to optimize DDPM. The classifier free guidance scale should be 0.0"
                unconditional_prob_cfg = 0.0

        assert self.num_timesteps_cond <= kwargs["timesteps"]

        conditioning_key = list(cond_stage_config.keys())

        self.conditioning_key = conditioning_key

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        
        self.optimize_ddpm_parameter = optimize_ddpm_parameter

        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.cond_stage_key_orig = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            print("[WARNING] scaling by std while training would results in an inaccurate estimation of the features' std. If you are training your own VAE, please use the script to estimate the features std before training")
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.unconditional_prob_cfg = unconditional_prob_cfg
        self.cond_stage_models = nn.ModuleList([])
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.conditional_dry_run_finished = False
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
        
    

    def initialize_optimizer_from_config(self, optimizer_config):
        # Configures the optimizer
        optimizer_target = optimizer_config['target']
        del optimizer_config['target']

        # model params
        params_list = list(self.model.parameters())
        for each in self.cond_stage_models:
            params_list = params_list + list(
                each.parameters()
            )  # Add the parameter from the conditional stage such as CLAP and dataset_id token embeddings

        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params_list.append(self.logvar)

        grad_parameters = [parameter for parameter in params_list if parameter.requires_grad]
        optimizer = optimizer_target(grad_parameters, **optimizer_config)

        # Configures the lr scheduler
        scheduler_frequency = self.global_config["lr_update_each_steps"]
        warmup_steps = self.warmup_steps

        if warmup_steps % scheduler_frequency != 0:
            raise ValueError(f"Warmup steps {warmup_steps} must be a multiple of the scheduler frequency {scheduler_frequency}")
        warmup_updates = warmup_steps // scheduler_frequency
        cosine_updates = self.global_config["max_steps"] // scheduler_frequency - warmup_updates
        final_lr = self.global_config["final_lr"]

        # Scheduler implementing the linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / warmup_updates, end_factor=1.0, total_iters=warmup_updates)
        # Scheduler to user after warmul
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_updates, eta_min=final_lr)
        # The composite scheduler
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_updates])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler, 
                "interval": "step",
                "frequency": scheduler_frequency
            }
        }


    def configure_optimizers(self):
        if self.optimizer_config is not None:
            return self.initialize_optimizer_from_config(self.optimizer_config)
        else:
            # Use ADAM by default without any scheduler
            lr = self.learning_rate
            params = list(self.model.parameters())

            # Train the conditional (i.e CLAP) model. 
            for each in self.cond_stage_models:
                params = params + list(
                    each.parameters()
                )  # Add the parameter from the conditional stage

            if self.learn_logvar:
                print("Diffusion model optimizing logvar")
                params.append(self.logvar)
            opt = torch.optim.AdamW(params, lr=lr)

        return {
            'optimizer': opt,
        }

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_factor == 1
            and self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"[INFO] setting scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def logsnr_cosine(self, timestamps, logsnr_min=-15, logsnr_max=15):
        t_min = np.arctan(np.exp(-0.5 * logsnr_max))
        t_max = np.arctan(np.exp(-0.5 * logsnr_min))
        logsnr_t = -2 * np.log(np.tan(t_min + timestamps * (t_max - t_min) ))
        return logsnr_t

    def logsnr_scheduler_cosine_shifted(self, timestamps, logsnr_min=-15, logsnr_max=15, ratio=1.0):
        # ration: between 0 and 1, a smaller ration means more noise.
        logsnr_t = self.logsnr_cosine(timestamps, logsnr_min, logsnr_max)
        logsnr_t += 2 * np.log(ratio)
        return logsnr_t

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        # INFO: an extensive hyper-parameter search for using simple linear or shifted_cosine scheduler resulted in no improvements on the overall performance
        if beta_schedule == 'simple_linear':
            print("[INFO] using simple linear scheduler")
            # override alphas for training and inference
            t = torch.linspace(0, 1, timesteps, dtype=torch.float64)
            alphas_cumprod = np.clip(1-t, a_min=1e-9, a_max=1.0) 
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

            (timesteps,) = alphas_cumprod.shape
            self.num_timesteps = int(timesteps)
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
            self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
            self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
            )

        elif beta_schedule == 'shifted_cosine':
            print("[INFO] using shifted coise scheduler")
            # override alphas for training and inference
            def sigmoid(x):
                return 1/(1 + np.exp(-x)) 
            
            timestamps = np.linspace(0, 1, timesteps, dtype=np.float64)
            logsnr_t = self.logsnr_scheduler_cosine_shifted(timestamps, logsnr_min=self.noise_scheduler_config['logsnr_min'],
                                                             logsnr_max=self.noise_scheduler_config['logsnr_max'],
                                                             ratio=self.noise_scheduler_config['shifted_cosine_low_ratio'])
            if 'shifted_cosine_high_ratio' in self.noise_scheduler_config \
                        and self.noise_scheduler_config['shifted_cosine_high_ratio'] != self.noise_scheduler_config['shifted_cosine_low_ratio']:
                logsnr_t_high = self.logsnr_scheduler_cosine_shifted(timestamps, logsnr_min=self.noise_scheduler_config['logsnr_min'],
                                                             logsnr_max=self.noise_scheduler_config['logsnr_max'],
                                                             ratio=self.noise_scheduler_config['shifted_cosine_high_ratio'])
                logsnr_t = timestamps * logsnr_t + (1 - timestamps) * logsnr_t_high
            alphas_cumprod = sigmoid(logsnr_t)
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

            (timesteps,) = alphas_cumprod.shape
            self.num_timesteps = int(timesteps)
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
            self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
            self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
            )

        else:
            super().register_schedule(
                given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
            )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}
        for i, cond_model_key in enumerate(config.keys()):
            model = instantiate_from_config(config[cond_model_key])
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }
            
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()

        # Classifier-free guidance
        if not unconditional_cfg:
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ](c)
        else:
            # when the cond_stage_key is "all", pick one random element out
            if isinstance(c, dict):
                c = c[list(c.keys())[0]]

            if isinstance(c, torch.Tensor):
                batchsize = c.size(0)
            elif isinstance(c, list):
                batchsize = len(c)
            else:
                raise NotImplementedError()

            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ].get_unconditional_condition(batchsize)

        return c

    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=True,
        return_decoding_output=False,
        return_encoder_input=False,
        return_encoder_output=False,
        unconditional_prob_cfg=0.1,
    ):
        x, encoder_posterior = None, None
        if return_first_stage_encode:
            x = super().get_input(batch, k)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
        else:
            z = None
        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False
            if self.conditional_dry_run_finished and self.make_decision(
                unconditional_prob_cfg
            ):
                unconditional_cfg = True
            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key][
                    "cond_stage_key"
                ]
                

                if cond_model_key in cond_dict.keys():
                    continue

                if not self.training:
                    if isinstance(
                        self.cond_stage_models[
                            self.cond_stage_model_metadata[cond_model_key]["model_idx"]
                        ],
                        CLAPAudioEmbeddingClassifierFreev2,
                    ) and cond_stage_key != 'text':
                        print(
                            f"Warning: CLAP model normally should use text for evaluation, currently using {cond_stage_key}"
                        )

                # The original data for conditioning
                # If cond_model_key is "all", that means the conditional model need all the information from a batch
                if cond_stage_key != "all":
                    xc = super().get_input(batch, cond_stage_key)
                    if type(xc) == torch.Tensor:
                        xc = xc.to(self.device)
                else:
                    xc = batch

                # if cond_stage_key is "all", xc will be a dictionary containing all keys
                # Otherwise xc will be an entry of the dictionary
                c = self.get_learned_conditioning(
                    xc, key=cond_model_key, unconditional_cfg=unconditional_cfg
                )

                # cond_dict will be used to condition the diffusion model
                # If one conditional model return multiple conditioning signal
                if isinstance(c, dict):
                    for k in c.keys():
                        cond_dict[k] = c[k]
                else:
                    cond_dict[cond_model_key] = c

        out = [z, cond_dict]

        if return_decoding_output:
            xrec = self.decode_first_stage(z)
            out += [xrec]

        if return_encoder_input:
            if x is None:
                x = super().get_input(batch, k)
                x = x.to(self.device)
            out += [x]

        if return_encoder_output:
            if x is None:
                x = super().get_input(batch, k)
                x = x.to(self.device)
            if encoder_posterior is None:
                encoder_posterior = self.encode_first_stage(x)
            out += [encoder_posterior]

        if not self.conditional_dry_run_finished:
            self.conditional_dry_run_finished = True

        # Output is a dictionary, where the value could only be tensor or tuple
        return out

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1.0 / self.scale_factor * z
            decoding = self.first_stage_model.decode(z)
        return decoding

    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform

    def encode_first_stage(self, x):
        with torch.no_grad():
            return self.first_stage_model.encode(x)

    def extract_possible_loss_in_cond_dict(self, cond_dict):
        # This function enable the conditional module to return loss function that can optimize them
        assert isinstance(cond_dict, dict)
        losses = {}

        for cond_key in cond_dict.keys():
            if "loss" in cond_key and "noncond" in cond_key:
                assert cond_key not in losses.keys()
                losses[cond_key] = cond_dict[cond_key]

        return losses

    def filter_useful_cond_dict(self, cond_dict):
        new_cond_dict = {}
        for key in cond_dict.keys():
            if key in self.cond_stage_model_metadata.keys():
                new_cond_dict[key] = cond_dict[key]

        # All the conditional key in the metadata should be used
        for key in self.cond_stage_model_metadata.keys():
            assert key in new_cond_dict.keys(), "%s, %s" % (
                key,
                str(new_cond_dict.keys()),
            )

        return new_cond_dict

    def shared_step(self, batch, **kwargs):
        if self.training:
            # Classifier-free guidance
            unconditional_prob_cfg = self.unconditional_prob_cfg
        else:
            unconditional_prob_cfg = 0.0  # TODO possible bug here

        x, c = self.get_input(
            batch, self.first_stage_key, unconditional_prob_cfg=unconditional_prob_cfg
        )

        if self.optimize_ddpm_parameter:
            loss, loss_dict = self(x, self.filter_useful_cond_dict(c), **kwargs)
        else:
            loss_dict = {}
            loss = None

        additional_loss_for_cond_modules = self.extract_possible_loss_in_cond_dict(c)
        assert isinstance(additional_loss_for_cond_modules, dict)

        loss_dict.update(additional_loss_for_cond_modules)

        if len(additional_loss_for_cond_modules.keys()) > 0:
            for k in additional_loss_for_cond_modules.keys():
                if loss is None:
                    loss = additional_loss_for_cond_modules[k]
                else:
                    loss = loss + additional_loss_for_cond_modules[k]

        if self.training:
            assert loss is not None

        return loss, loss_dict

    def forward(self, x, c, t_range=None, *args, **kwargs):
        if t_range is not None:
            t = torch.randint(
                t_range[0], t_range[1], (x.shape[0],), device=self.device
            ).long()
        else:
            t = torch.randint(
                0, self.num_timesteps, (x.shape[0],), device=self.device
            ).long()

        loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs)
        return loss, loss_dict

    def reorder_cond_dict(self, cond_dict):
        # To make sure the order is correct
        new_cond_dict = {}
        for key in self.conditioning_key:
            new_cond_dict[key] = cond_dict[key]
        return new_cond_dict

    def apply_model(self, x_noisy, t, cond, return_model_out = False, return_ids=False, **kwargs):
        cond = self.reorder_cond_dict(cond)

        x_recon = self.model(x_noisy, t, cond_dict=cond, return_model_out=return_model_out, **kwargs)

        if return_model_out:
            return x_recon
        elif isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None):
        target_dims = [1, 2, 3]
        if len(x_start.shape) == 3:
            target_dims = [1, 2]
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean(target_dims)
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        if self.original_elbo_weight != 0:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=target_dims)
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
            loss += self.original_elbo_weight * loss_vlb
            loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        return_model_out=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids, return_model_out=return_model_out, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if return_codebook_ids:
            model_out, logits = model_out

        if return_model_out:
            model_out, model_log = model_out
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_model_out:
            return model_mean, posterior_variance, posterior_log_variance, model_log
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        return_model_out=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            return_model_out=return_model_out,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            **kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
        elif return_model_out: 
            model_mean, _, model_log_variance, model_out = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )

        if return_model_out:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                model_out,
            )
        
        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # Not used in this class
    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive Generation",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        prev_iter_model_out=None
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            if self.backbone_type.lower() == 'fit':
                # FIT needs the latent tokens from preious step for warm initalization
                img, prev_iter_model_out = self.p_sample(
                    img,
                    cond,
                    ts,
                    clip_denoised=self.clip_denoised,
                    quantize_denoised=quantize_denoised,
                    prev_iter_model_out=prev_iter_model_out,
                    return_model_out=True)
            else:
                img = self.p_sample(
                    img,
                    cond,
                    ts,
                    clip_denoised=self.clip_denoised,
                    quantize_denoised=quantize_denoised)

            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.latent_t_size, self.latent_f_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
            **kwargs,
        )

    def save_waveform(self, waveform, savepath, name="outwav"):
        print(f"[INFO] saving output samples of shape {waveform.shape} at {savepath}")
        waveform_save_paths = []
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            todo_waveform = waveform[i, 0]
            todo_waveform = (
                todo_waveform / np.max(np.abs(todo_waveform))
            ) * 0.8  # Normalize the energy of the generation output
            sf.write(path, todo_waveform, samplerate=self.sampling_rate)
            waveform_save_paths.append(path)
        return waveform_save_paths

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if ddim and not use_plms:
            # Normally, ddim sampler is used. The other sampler do not support the FIT backbone
            ddim_sampler = DDIMSampler(self)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                backbone_type=self.backbone_type,
                **kwargs,
            )
        elif use_plms:
            assert self.backbone_type.lower() != 'fit', "[ERROR] PLMS is not supported yet for the FIT backbone"
            plms_sampler = PLMSSampler(self)
            samples, intermediates = plms_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        else:
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        print("[INFO] Finished sampling:", samples.shape)
        return samples, intermediate
    
    @torch.no_grad()
    def log_spectrograms(self, spectrograms, name, step, grid=(2, 2)):
        # Assuming spectrograms is a PyTorch tensor of shape [B, 1, H, W]
        spectrograms = spectrograms.cpu().detach().numpy()

        fig, axs = plt.subplots(grid[0], grid[1], figsize=(10, 10))  
        axs = axs.flatten() 

        for idx, ax in enumerate(axs):
            if idx < len(spectrograms):
                spec = spectrograms[idx].squeeze()  
                img = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            else:
                ax.axis('off')  
            ax.set_title(f'Spectrogram {idx}')

        plt.colorbar(img, ax=axs, orientation='horizontal', pad=0.05)  # Add a colorbar to the last spectrogram
        
        # Save the figure to a buffer (avoiding saving and reading from disk)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        image = Image.open(buf)
        if self.logger is not None:
            self.logger.experiment.log({name: [wandb.Image(image, caption="Log Mel Spectrograms")]}, step=step)
        
        plt.close(fig)  # Close the figure to free memory
    

    

    @torch.no_grad()
    def filename_to_seed(self, fname):
        """Convert a filename to a consistent seed."""
        return int(hashlib.md5(fname.encode()).hexdigest(), 16)

    @torch.no_grad()
    def generate_noise_for_batch(self, batch, C, H, W, n_gen):
        """Generate consistent noise for a batch based on file names."""
        B = len(batch["fname"])  # Assuming batch["fname"] is a list of filenames
        noise_batch = torch.empty(B*n_gen, C, H, W) #if W > 1 else  torch.empty(B*n_gen, C, H)# Pre-allocate noise tensor
        
        for i, fname in enumerate(batch["fname"]):
            seed = self.filename_to_seed(fname) % (2**32)  # Reduce seed to 32-bit integer range
            generator = torch.Generator().manual_seed(seed)  # Create a new generator for this seed
            
            # Generate noise for this sample
            for j in range(n_gen):
                noise = torch.randn(C, H, W, generator=generator)
                noise_batch[i*n_gen+j] = noise
        
        return noise_batch

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        name = self.name
        unconditional_guidance_scale = self.unconditional_guidance_scale
        ddim_sampling_steps = self.ddim_sampling_steps
        n_gen = self.n_gen   

        return self.generate_sample([batch], name=name, unconditional_guidance_scale=unconditional_guidance_scale, ddim_sampling_steps=ddim_sampling_steps, n_gen=n_gen)
    
    
    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name=None,
        use_plms=False,
        limit_num=None,
        use_ema=True,
        **kwargs,
    ):
        # Generate n_gen times and select the best
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        if name is None:
            name = self.get_validation_folder_name()

        waveform_save_path = os.path.join(self.get_log_dir(), name)
        waveform_save_path = waveform_save_path.replace("val_0", "infer")

        os.makedirs(waveform_save_path, exist_ok=True)

        with self.ema_scope("Plotting", use_ema=use_ema):
            for i, batch in enumerate(batchs):
                fnames = list(super().get_input(batch, "fname"))
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
                    return_first_stage_encode=False,
                )

                if limit_num is not None and i * z.size(0) > limit_num:
                    break

                c = self.filter_useful_cond_dict(c)
                text = super().get_input(batch, "text")

                # Generate multiple samples
                num_samples = len(batch['text']) 
                batch_size = len(batch['text']) * n_gen

                # Generate multiple samples at a time and filter out the best
                # The condition to the diffusion wrapper can have many format
                for cond_key in c.keys():
                    if isinstance(c[cond_key], list):
                        for i in range(len(c[cond_key])):
                            c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)
                    elif isinstance(c[cond_key], dict):
                        for k in c[cond_key].keys():
                            c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)
                    else:
                        c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)

                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = {}
                    for key in self.cond_stage_model_metadata:
                        model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                        unconditional_conditioning[key] = self.cond_stage_models[
                            model_idx
                        ].get_unconditional_condition(batch_size)

                # Prepare X_T
                # shape = (batch_size, self.channels, self.latent_t_size, self.latent_f_size)
                x_T = self.generate_noise_for_batch(batch, self.channels, self.latent_t_size, self.latent_f_size, n_gen=n_gen).to(self.device) 

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )

                mel = self.decode_first_stage(samples)

                if self.log_melspectrogran:
                    mel_grid = make_grid(mel, nrows=2) # TODO: decide on the number of rows
                    self.log_spectrograms(mel[:4].exp(), "val/mel_spectrogram", self.global_step)
                    self.logger.experiment.log({"val/mel_spectrogram": [wandb.Image(mel_grid.permute(1, 2, 0).detach().cpu().numpy(), caption="Spectrograms")]}, step=self.global_step, commit=False)
                
                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )
                all_clap_sim = None
                best_clap_sim = None
                if n_gen > 1:
                    best_index = []
                    similarity = self.clap.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                    )
                    all_clap_sim = similarity if all_clap_sim is None else torch.cat([all_clap_sim, similarity])
                    for i in range(num_samples):
                        candidates = similarity[i :: num_samples]
                        max_index = torch.argmax(candidates).item()
                        best_index.append(i + max_index * num_samples)

                    best_clap_sim = similarity[best_index] if best_clap_sim is None else torch.cat([best_clap_sim, similarity[best_index]])
                    waveform = waveform[best_index]
                
                self.save_waveform(waveform, waveform_save_path, name=fnames)
        return waveform_save_path, all_clap_sim, best_clap_sim
    
    
    @torch.no_grad()
    def text_to_audio_batch(
        self,
        batch,
        waveform_save_dir='samples/model_output',
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name=None,
        use_plms=False,
        limit_num=None,
        use_ema=True,
        **kwargs,
    ):
        os.makedirs(waveform_save_dir, exist_ok=True)
        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None

        with self.ema_scope("Plotting", use_ema=use_ema):
                fnames = list(batch["fname"])
                _, c = self.get_input(
                    batch,
                    self.first_stage_key, # fbank
                    unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
                    return_first_stage_encode=False,
                )

                c = self.filter_useful_cond_dict(c)
                text = batch['text']

                # Generate multiple samples
                num_samples = len(batch['text']) 
                batch_size = len(batch['text']) * n_gen

                # Generate multiple samples at a time and filter out the best
                # The condition to the diffusion wrapper can have many format
                for cond_key in c.keys():
                    if isinstance(c[cond_key], list):
                        for i in range(len(c[cond_key])):
                            c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)
                    elif isinstance(c[cond_key], dict):
                        for k in c[cond_key].keys():
                            c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)
                    else:
                        c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)

                text = text * n_gen
                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = {}
                    for key in self.cond_stage_model_metadata:
                        model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                        unconditional_conditioning[key] = self.cond_stage_models[
                            model_idx
                        ].get_unconditional_condition(batch_size)

                # Prepare X_T
                # shape = (batch_size, self.channels, self.latent_t_size, self.latent_f_size)
                x_T = self.generate_noise_for_batch(batch, self.channels, self.latent_t_size, self.latent_f_size, n_gen=n_gen).to(self.device) 

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_dir, bs=None, name=fnames, save=False
                )
                if n_gen > 1:
                    best_index = []
                    similarity = self.clap.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                    )
                    for i in range(num_samples):
                        candidates = similarity[i :: num_samples]
                        max_index = torch.argmax(candidates).item()
                        best_index.append(i + max_index * num_samples)
                        
                    waveform = waveform[best_index]
                
                waveform_save_paths = self.save_waveform(waveform, waveform_save_dir, name=fnames)
        return waveform_save_paths
    
    @torch.no_grad()
    def text_to_audio(self, prompt, dataset_name='audiocaps', fname=None, *args, **kwargs):
        if fname is None:
            fname = prompt.replace(" ", "_").replace("'", "_").replace('"', "_")
        return self.text_to_audio_batch({"text":[prompt], "dataset_name":[dataset_name], "fname":[fname]}, *args, **kwargs)[0]
    
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key, backbone_type='unet', scale_by_std=False):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

        self.conditioning_key = conditioning_key
        self.scale_by_std = scale_by_std

        for key in self.conditioning_key:
            if (
                "concat" in key
                or "crossattn" in key
                or "hybrid" in key
                or "film" in key
                or "noncond" in key
                or 'ignore' in key
            ):
                continue
            else:
                raise Value("The conditioning key %s is illegal" % key)

        self.being_verbosed_once = False
        self.backbone_type = backbone_type

    def forward(self, x, t, cond_dict: dict = {}, return_model_out=False, prev_iter_model_out=None):
        x = x.contiguous()
        # scale by std, for the simple linear scheduler
        if self.scale_by_std:
            target_dims = [1, 2, 3]
            if len(x.shape) == 3:
                target_dims = [1, 2]
            std = x.std(target_dims, keepdim=True)
            print("Estimated STD is", std)
            x = x / x.std(target_dims, keepdim=True)
        t = t.contiguous()

        # x with condition (or maybe not)
        xc = x

        y = None
        context_list, attn_mask_list = [], []

        conditional_keys = cond_dict.keys()
        for key in conditional_keys:
            if "concat" in key:
                xc = torch.cat([x, cond_dict[key].unsqueeze(1)], dim=1)
            elif "film" in key:
                if y is None:
                    y = cond_dict[key].squeeze(1)
                else:
                    y = torch.cat([y, cond_dict[key].squeeze(1)], dim=-1)
            elif "crossattn" in key:
                # assert context is None, "You can only have one context matrix, got %s" % (cond_dict.keys())
                if isinstance(cond_dict[key], dict):
                    for k in cond_dict[key].keys():
                        if "crossattn" in k:
                            context, attn_mask = cond_dict[key][
                                k
                            ]  # crossattn_audiomae_pooled: torch.Size([12, 128, 768])
                else:
                    assert len(cond_dict[key]) == 2, (
                        "The context condition for %s you returned should have two element, one context one mask"
                        % (key)
                    )
                    context, attn_mask = cond_dict[key]

                # The input to the UNet model is a list of context matrix
                context_list.append(context)
                attn_mask_list.append(attn_mask)

            elif (
                "noncond" in key
            ):  # If you use loss function in the conditional module, include the keyword "noncond" in the return dictionary
                continue
            else:
                raise NotImplementedError()


        if self.backbone_type.lower() == 'fit':
            data_entries = DataEntries()
            data_entries.add(DataEntry("input", xc, type="input"))
            data_entries.add(DataEntry("summary_text_embeddings", y.unsqueeze(1)))
            data_entries.add(DataEntry("noise_labels", t))
            data_entries.add(DataEntry("dataset_id", cond_dict['noncond_dataset_ids']))

            # Add data entries for dataset ID. (torch.int64)            
            if prev_iter_model_out is not None: # add latents back to input for self-conditioning
                for current_input_key in list(prev_iter_model_out.keys(types_filter="self-conditioning")):
                        data_entries.add(prev_iter_model_out[current_input_key].shallow_copy())

            out = self.diffusion_model(data_entries)
            if return_model_out:
                return (out['input'].data, out)
            else:
                return out['input'].data
        
        else:
            out = self.diffusion_model(
                xc, t, context_list=context_list, y=y, context_attn_mask_list=attn_mask_list
            )
        return out


if __name__ == "__main__":
    from src.tools.configuration import Configuration
    
    model_config = "settings/simple_runs/genau.yaml"
    configuration = Configuration(model_config)
    model_config = configuration.get_config()
    
    latent_diffusion = GenAu(ckpt_path='data/pretrained/checkpoint-fad-133.00-global_step=103499.ckpt',
                             **model_config["model"]["params"]).cuda()
    
    print("[INFO] saved audio sample", latent_diffusion.text_to_audio(
        prompt="A gun cocking then firing as metal clanks on a hard surface followed by a man talking during an electronic laser effect as gunshots and explosions go off in the distance",
        unconditional_guidance_scale=4.0,
        n_gen=3,
        use_ema=False))