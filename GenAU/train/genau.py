# Author: Moayed Haji Ali
# Email: mh155@rice.edu
# Date: 8 June 2024

# based on code from 
# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import shutil
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_START_METHOD'] = 'thread'

import argparse
from tqdm import tqdm
import logging


import torch
from torch.utils.data import DataLoader
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer, seed_everything

from src.utilities.model.model_checkpoint import S3ModelCheckpoint
from src.tools.logger import Logger
from src.tools.configuration import Configuration
from src.tools.training_utils import (
    get_restore_step,
    copy_test_subset_data,
)
from src.utilities.model.model_util import instantiate_from_config
from src.utilities.data.videoaudio_dataset import VideoAudioDataset, custom_collate_fn
from src.tools.download_manager import get_checkpoint_path
logging.basicConfig(level=logging.WARNING)


def main(configs, config_yaml_path, exp_group_name, exp_name, debug=False):
    if "seed" in configs.keys():
        print(f"[INFO] SEED EVERYTHING TO {configs['seed']}")
        seed_everything(configs["seed"])
    else:
        print("[INFO] SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs['training'].keys():
        torch.set_float32_matmul_precision(
            configs['training']["precision"]
        )  # highest, high, medium

    log_path = configs['logging']["log_directory"]
    batch_size = configs["model"]["params"]["batchsize"]
    print(f"[INFO] using batch size of {batch_size}")

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    augment_p = configs['data']['augment_p'] if 'augment_p' in configs['data'] else 0.0
    dataset = VideoAudioDataset(configs, split="train", add_ons=dataloader_add_ons, load_video=False, load_audio=True, sample_single_caption=True, augment_p=augment_p)
    
    print("using batch size of:", batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=configs['data'].get('num_workers', 32),
        pin_memory=True,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    val_dataset = VideoAudioDataset(configs, split="test", add_ons=dataloader_add_ons, load_video=False, load_audio=True, sample_single_caption=True)

    val_loader = DataLoader(
        val_dataset,
        num_workers=configs['data'].get('num_workers', 32),
        batch_size=max(1, batch_size // configs['model']['params']['evaluation_params']['n_candidates_per_samples']),
        collate_fn=custom_collate_fn
    )

    # Copy test data
    test_data_subset_folder = os.path.join(
        os.path.dirname(configs['logging']["log_directory"]),
        "testset_data",
        val_dataset.dataset_name,
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    copy_test_subset_data(val_dataset, test_data_subset_folder)

    config_reload_from_ckpt = configs.get("reload_from_ckpt", None)
    limit_val_batches = configs["step"].get("limit_val_batches", None)
    limit_train_batches = configs["step"].get("limit_train_batches", None)
    validation_every_n_epochs = configs["step"].get("validation_every_n_epochs", None)
    val_check_interval = configs["step"].get("val_check_interval", None)
    max_steps = configs["step"]["max_steps"]
    save_top_k = configs["logging"]["save_top_k"]
    save_checkpoint_every_n_steps = configs["logging"]["save_checkpoint_every_n_steps"]

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    checkpoint_callback = S3ModelCheckpoint(
        bucket_name=configs['logging'].get('S3_BUCKET', None),
        s3_folder=configs['logging'].get('S3_FOLDER', None),
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=False,
    )

    os.makedirs(checkpoint_path, exist_ok=True)
    
    config_copy_dir = os.path.join(wandb_path, 'config')
    os.makedirs(config_copy_dir, exist_ok=True)
    shutil.copy(config_yaml_path, config_copy_dir)

    is_external_checkpoints = False
    if len(os.listdir(checkpoint_path)) > 0 and "resume_training" in configs['training'] and configs['training']["resume_training"]:
        print("[INFO] Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("[INFO] Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        is_external_checkpoints = True
        print("[INFO] Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("[INFO] Training from scratch")
        resume_from_checkpoint = None

    devices = torch.cuda.device_count()

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)


    wandb_logger = Logger(
        config=configs, 
        checkpoints_directory=wandb_path,
        run_name="%s/%s" % (exp_group_name, exp_name),
        offline=debug
    ).get_logger()

    latent_diffusion.test_data_subset_path = test_data_subset_folder

    print("[INFO] ==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("[INFO] ==> Perform validation every %s epochs" % validation_every_n_epochs)

    nodes_count = configs['training']["nodes_count"]
    if nodes_count == -1:
        if "WORLD_SIZE" in os.environ:
            nodes_count = int(os.environ["WORLD_SIZE"]) // torch.cuda.device_count()
        else:
            nodes_count = 1

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        max_steps=max_steps,
        num_sanity_val_steps=1,
        num_nodes=nodes_count,
        limit_val_batches=limit_val_batches,
        limit_train_batches=limit_train_batches,
        check_val_every_n_epoch=validation_every_n_epochs,
        val_check_interval=val_check_interval,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
        gradient_clip_val=configs["model"]["params"].get("clip_grad", None)
    )
    

    if is_external_checkpoints:
        if resume_from_checkpoint is not None:
            ckpt = torch.load(resume_from_checkpoint)["state_dict"]

            key_not_in_model_state_dict = []
            size_mismatch_keys = []
            state_dict = latent_diffusion.state_dict()
            print("[INFO] Filtering key for reloading:", resume_from_checkpoint)
            print(
                "[INFO] State dict key size:",
                len(list(state_dict.keys())),
                len(list(ckpt.keys())),
            )
            for key in tqdm(list(ckpt.keys())):
                if key not in state_dict.keys():
                    key_not_in_model_state_dict.append(key)
                    del ckpt[key]
                    continue
                if state_dict[key].size() != ckpt[key].size():
                    del ckpt[key]
                    size_mismatch_keys.append(key)

            latent_diffusion.load_state_dict(ckpt, strict=False)

        trainer.fit(latent_diffusion, loader, val_loader)
    else:
        trainer.fit(
            latent_diffusion, loader, val_loader, ckpt_path=resume_from_checkpoint
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="debug mode",
    )

    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )
    
    parser.add_argument(
        "--dataset_keys",
        nargs='*',
        required=False,
        default=[],
        help="A list of dataset keys for training or finetuning",
    )
    
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    exp_name = os.path.basename(args.config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(args.config_yaml))
    
    configuration = Configuration(args.config_yaml)
    config_yaml = configuration.get_config()

    if args.dataset_keys:
        config_yaml['data']['train'] = args.dataset_keys
    
    if args.reload_from_ckpt is not None and not os.path.exists(args.reload_from_ckpt):
        args.reload_from_ckpt = get_checkpoint_path(args.reload_from_ckpt)
        
    if args.reload_from_ckpt is not None:
        config_yaml['model']['params']['ckpt_path'] = args.reload_from_ckpt

    main(config_yaml, args.config_yaml, exp_group_name, exp_name, debug=args.debug)
