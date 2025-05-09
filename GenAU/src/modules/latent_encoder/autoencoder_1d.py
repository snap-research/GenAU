from email.policy import strict
import torch
import os

import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from src.modules.diffusionmodules.ema import *

from src.modules.diffusionmodules.model_1d import Encoder1D, Decoder1D
from src.modules.diffusionmodules.distributions import (
    DiagonalGaussianDistribution,
)

import wandb
from src.utilities.model.model_util import instantiate_from_config
import soundfile as sf

from src.utilities.model.model_util import get_vocoder
from src.tools.training_utils import synth_one_sample
from src.tools.download_manager import get_checkpoint_path
import itertools


class AutoencoderKL1D(pl.LightningModule):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        batchsize=None,
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        sampling_rate=16000,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        image_key="fbank",
        colorize_nlabels=None,
        monitor=None,
        ckpt_root='data/checkpoints',
        base_learning_rate=1e-5,
    ):
        super().__init__()
        self.automatic_optimization = False
        assert (
            "mel_bins" in ddconfig.keys()
        ), "mel_bins is not specified in the Autoencoder config"
        num_mel = ddconfig["mel_bins"]
        self.image_key = image_key
        self.sampling_rate = sampling_rate
        self.encoder = Encoder1D(**ddconfig)
        self.decoder = Decoder1D(**ddconfig)

        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        else:
            self.loss = None
        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv1d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)

        if self.image_key == "fbank":
            self.vocoder = get_vocoder(None, "cpu", num_mel, ROOT=ckpt_root)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.learning_rate = float(base_learning_rate)
        print("[INFO] Initial learning rate %s" % self.learning_rate)

        self.time_shuffle = time_shuffle
        
        if reload_from_ckpt is not None and not os.path.exists(reload_from_ckpt):
            reload_from_ckpt = get_checkpoint_path(reload_from_ckpt)
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.feature_cache = None
        self.flag_first_run = True
        self.train_step = 0

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None

        if not self.reloaded and self.reload_from_ckpt is not None:
            
            print("[INFO] --> Reload weight of autoencoder from %s" % self.reload_from_ckpt)
            checkpoint = torch.load(self.reload_from_ckpt)

            load_todo_keys = {}
            pretrained_state_dict = checkpoint["state_dict"]
            current_state_dict = self.state_dict()
            for key in current_state_dict:
                if (
                    key in pretrained_state_dict.keys()
                    and pretrained_state_dict[key].size()
                    == current_state_dict[key].size()
                ):
                    load_todo_keys[key] = pretrained_state_dict[key]
                else:
                    print("[INFO] Key %s mismatch during loading, seems fine" % key)

            self.load_state_dict(load_todo_keys, strict=False)
            self.reloaded = True
        else:
            print("[INFO] Training from scratch")

    def get_log_dir(self):
        return os.path.join(
            self.logger_save_dir, self.logger_exp_group_name, self.logger_exp_name
        )

    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_name = exp_name
        self.logger_exp_group_name = exp_group_name

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("[INFO] Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"[INFO] Restored from {path}")

    def encode(self, x):
        x = self.freq_split_subband(x) # B x 1 x L x C

        # convert to 1D input 
        x = x.squeeze(1).permute(0, 2, 1) # B x C x L
        h = self.encoder(x) # 
        moments = self.quant_conv(h) 
        posterior = DiagonalGaussianDistribution(moments.unsqueeze(-1)) # conver to 4D: B x C x H x W
        return posterior

    def decode(self, z):
        z = z.squeeze(-1) # expect 4D input: B x C x H x W
        z = self.post_quant_conv(z)
        dec = self.decoder(z) # B x C x L
        
        # convert back to an image shape
        dec = dec.unsqueeze(1).permute(0, 1, 3, 2) # B x 1 X L x C

        dec = self.freq_merge_subband(dec)
        return dec

    def decode_to_waveform(self, dec):
        from src.utilities.model.model_util import vocoder_infer

        if self.image_key == "fbank":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = vocoder_infer(dec, self.vocoder)
        elif self.image_key == "stft":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = self.wave_decoder(dec)
        return wav_reconstruction

    def visualize_latent(self, input):
        import matplotlib.pyplot as plt

        np.save("input.npy", input.cpu().detach().numpy())
        time_input = input.clone()
        time_input[:, :, :, :32] *= 0
        time_input[:, :, :, :32] -= 11.59

        np.save("time_input.npy", time_input.cpu().detach().numpy())

        posterior = self.encode(time_input)
        latent = posterior.sample()
        np.save("time_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("freq_%s.png" % i)
            plt.close()

        freq_input = input.clone()
        freq_input[:, :, :512, :] *= 0
        freq_input[:, :, :512, :] -= 11.59

        np.save("freq_input.npy", freq_input.cpu().detach().numpy())

        posterior = self.encode(freq_input)
        latent = posterior.sample()
        np.save("freq_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("time_%s.png" % i)
            plt.close()

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if self.flag_first_run:
            print("[INFO] Latent size: ", z.size())
            self.flag_first_run = False

        dec = self.decode(z)

        return dec, posterior

    def get_input(self, batch):
        fname, text, waveform, stft, fbank = (
            batch["fname"],
            batch["text"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
        )

        ret = {}

        ret["fbank"], ret["stft"], ret["fname"], ret["waveform"] = (
            fbank.unsqueeze(1),
            stft.unsqueeze(1),
            fname,
            waveform.unsqueeze(1),
        )

        return ret

    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size(-1) % self.subband == 0
        assert ch == 1

        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        inputs_dict = self.get_input(batch)
        inputs = inputs_dict[self.image_key]
        waveform = inputs_dict["waveform"]

        if batch_idx % 5000 == 0 and self.local_rank == 0:
            print("Log train image")
            self.log_images(inputs, waveform=waveform)

        reconstructions, posterior = self(inputs)

        if self.image_key == "stft":
            rec_waveform = self.decode_to_waveform(reconstructions)
        else:
            rec_waveform = None

        # train the discriminator
        # If working on waveform, inputs is STFT, reconstructions are the waveform
        # If working on the melspec, inputs is melspec, reconstruction are also mel spec
        discloss, log_dict_disc = self.loss(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posterior,
            waveform=waveform,
            rec_waveform=rec_waveform,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        self.log(
            "discloss",
            discloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        d_opt.zero_grad()
        self.manual_backward(discloss)
        d_opt.step()

        self.log(
            "train_step",
            self.train_step,
            prog_bar=False,
            logger=False,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        aeloss, log_dict_ae = self.loss(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posterior,
            waveform=waveform,
            rec_waveform=rec_waveform,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "posterior_std",
            torch.mean(posterior.var),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr",
            lr,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        self.log_dict(
            log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.train_step += 1
        g_opt.zero_grad()
        self.manual_backward(aeloss)
        g_opt.step()

    def validation_step(self, batch, batch_idx):
        inputs_dict = self.get_input(batch)
        inputs = inputs_dict[self.image_key]
        waveform = inputs_dict["waveform"] # B x 1 x 1 x L
        if batch_idx <= 3:
            print("Log val image")
            self.log_images(inputs, train=False, waveform=waveform)

        reconstructions, posterior = self(inputs)

        if self.image_key == "stft":
            rec_waveform = self.decode_to_waveform(reconstructions)
        else:
            rec_waveform = None

        aeloss, log_dict_ae = self.loss(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posterior,
            waveform=waveform,
            rec_waveform=rec_waveform,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posterior,
            waveform=waveform,
            rec_waveform=rec_waveform,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        inputs_dict = self.get_input(batch)
        inputs = inputs_dict[self.image_key]
        waveform = inputs_dict["waveform"]
        fnames = inputs_dict["fname"]

        reconstructions, posterior = self(inputs)
        save_path = os.path.join(
            self.get_log_dir(), "autoencoder_result_audiocaps", str(self.global_step)
        )

        if self.image_key == "stft":
            wav_prediction = self.decode_to_waveform(reconstructions)
            wav_original = waveform
            self.save_wave(
                wav_prediction, fnames, os.path.join(save_path, "stft_wav_prediction")
            )
        else:
            wav_vocoder_gt, wav_prediction = synth_one_sample(
                inputs.squeeze(1),
                reconstructions.squeeze(1),
                labels="validation",
                vocoder=self.vocoder,
            )
            self.save_wave(
                wav_vocoder_gt, fnames, os.path.join(save_path, "fbank_vocoder_gt_wave")
            )
            self.save_wave(
                wav_prediction, fnames, os.path.join(save_path, "fbank_wav_prediction")
            )

    def save_wave(self, batch_wav, fname, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for wav, name in zip(batch_wav, fname):
            name = os.path.basename(name)

            sf.write(os.path.join(save_dir, name), wav, samplerate=self.sampling_rate)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
        )

        if self.image_key == "stft":
            params += list(self.wave_decoder.parameters())

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        if self.image_key == "fbank":
            disc_params = self.loss.discriminator.parameters()
        elif self.image_key == "stft":
            disc_params = itertools.chain(
                self.loss.msd.parameters(), self.loss.mpd.parameters()
            )

        opt_disc = torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, train=True, only_inputs=False, waveform=None, **kwargs):
        log = dict()
        x = batch.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(posterior.sample())
            log["reconstructions"] = xrec

        log["inputs"] = x
        wavs = self._log_img(log, train=train, index=0, waveform=waveform)
        return wavs

    def _log_img(self, log, train=True, index=0, waveform=None):
        images_input = self.tensor2numpy(log["inputs"][index, 0]).T
        images_reconstruct = self.tensor2numpy(log["reconstructions"][index, 0]).T
        images_samples = self.tensor2numpy(log["samples"][index, 0]).T

        if train:
            name = "train"
        else:
            name = "val"

        if self.logger is not None:
            self.logger.log_image(
                "img_%s" % name,
                [images_input, images_reconstruct, images_samples],
                caption=["input", "reconstruct", "samples"],
            )

        inputs, reconstructions, samples = (
            log["inputs"],
            log["reconstructions"],
            log["samples"],
        )

        if self.image_key == "fbank":
            wav_original, wav_prediction = synth_one_sample(
                inputs[index],
                reconstructions[index],
                labels="validation",
                vocoder=self.vocoder,
            )
            wav_original, wav_samples = synth_one_sample(
                inputs[index], samples[index], labels="validation", vocoder=self.vocoder
            )
            wav_original, wav_samples, wav_prediction = (
                wav_original[0],
                wav_samples[0],
                wav_prediction[0],
            )
        elif self.image_key == "stft":
            wav_prediction = (
                self.decode_to_waveform(reconstructions)[index, 0]
                .cpu()
                .detach()
                .numpy()
            )
            wav_samples = (
                self.decode_to_waveform(samples)[index, 0].cpu().detach().numpy()
            )
            wav_original = waveform[index, 0].cpu().detach().numpy()

        if self.logger is not None:
            self.logger.experiment.log(
                {
                    "original_%s"
                    % name: wandb.Audio(
                        wav_original, caption="original", sample_rate=self.sampling_rate
                    ),
                    "reconstruct_%s"
                    % name: wandb.Audio(
                        wav_prediction,
                        caption="reconstruct",
                        sample_rate=self.sampling_rate,
                    ),
                    "samples_%s"
                    % name: wandb.Audio(
                        wav_samples, caption="samples", sample_rate=self.sampling_rate
                    ),
                }
            )

        return wav_original, wav_prediction, wav_samples

    def tensor2numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, ddconfig=None, image_key='fbank', ckpt_root="data/checkpoints", **kwargs):
        super().__init__()
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        self.image_key = image_key
        num_mel = ddconfig["mel_bins"]
        self.vocoder = get_vocoder(None, "cpu", num_mel, ROOT=ckpt_root)

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def decode_to_waveform(self, dec):
        from src.utilities.model.model_util import vocoder_infer
        if self.image_key == "fbank":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = vocoder_infer(dec, self.vocoder)
        elif self.image_key == "stft":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = self.wave_decoder(dec)
        return wav_reconstruction
    
    def forward(self, x, *args, **kwargs):
        return x
