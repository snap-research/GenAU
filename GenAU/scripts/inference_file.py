import shutil
import os

import argparse
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning import seed_everything

from src.tools.training_utils import get_restore_step
from src.utilities.model.model_util import instantiate_from_config
from src.tools.training_utils import build_dataset_json_from_list
from src.tools.configuration import Configuration
from src.utilities.data.videoaudio_dataset import VideoAudioDataset, custom_collate_fn
from src.tools.download_manager import get_checkpoint_path

@torch.no_grad()
def infer(dataset_json, configs, config_yaml_path, exp_group_name, exp_name, seed=0, 
         n_cand=1, cfg_weight=3.5, ddim_steps=200, strategy='wo_ema'):
    seed_everything(seed)

    if "precision" in configs['training'].keys():
        torch.set_float32_matmul_precision(
            configs['training']["precision"]
        )  # highest, high, medium


    log_path = configs['logging']["log_directory"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    val_dataset = VideoAudioDataset(
        config=configs, 
        split='test',
        add_ons=dataloader_add_ons,
        dataset_json=dataset_json,
        load_audio=False,
        load_video=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=42,
        collate_fn=custom_collate_fn
    )

    config_reload_from_ckpt = configs.get("reload_from_ckpt", None)
    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")
    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)

    if config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    elif len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    else:
        raise "Please specify a pre-trained checkpoint"

    configs['model']['params']['ckpt_path'] = resume_from_checkpoint
    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()
        
    latent_diffusion.generate_sample(
        val_loader,
        unconditional_guidance_scale=cfg_weight,
        ddim_steps=ddim_steps,
        n_gen=n_cand,
        use_ema=(strategy != 'wo_ema')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        default=None,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='genau-l-full-hq-data',
        required=False,
        help="Model name",
    )
    
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0
    )
    
    parser.add_argument(
        "-cfg",
        "--cfg_weight",
        type=float,
        default=4.0
    )
    
    parser.add_argument(
        "--n_cand",
        type=int,
        default=3,
        help="number of candidates for clap reranking"
    )
    
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim steps for sampling"
    )
    
    parser.add_argument(
        "-l",
        "--list_inference",
        type=str,
        default=None,
        required=True,
        help="The filelist that contain captions (and optionally filenames)",
    )
    parser.add_argument(
        "-ckpt",
        "--reload_from_ckpt",
        type=str,
        required=False,
        help="the checkpoint path for the model. If not provided, the most recent checkpoint from the log folder of the provided caption will be used",
    )

    
    parser.add_argument(
        "--strategy",
        type=str,
        required=False,
        default='ema',
        help="The strategy of combining weights from different checkpoint: wo_ema, avg_ckpt, or ema",
    )

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    if args.config_yaml is None:
        args.config_yaml = get_checkpoint_path(f"{args.model}_config")
    if args.reload_from_ckpt is None:
        args.reload_from_ckpt = get_checkpoint_path(args.model)
        
    config_yaml = args.config_yaml
    if args.list_inference is not None:
        dataset_json = build_dataset_json_from_list(args.list_inference)
    else:
        dataset_json = None
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    configuration = Configuration(config_yaml)
    configs = configuration.get_config()


    if args.reload_from_ckpt != None:
        configs["reload_from_ckpt"] = args.reload_from_ckpt

    infer(dataset_json=dataset_json, 
          configs=configs, 
          config_yaml_path=args.config_yaml, 
          exp_group_name=exp_group_name,
          exp_name=exp_name, 
          seed=args.seed,
          n_cand=args.n_cand,
          ddim_steps=args.ddim_steps,
          cfg_weight=args.cfg_weight,
          strategy=args.strategy)
