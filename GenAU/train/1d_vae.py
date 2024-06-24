# Author: Moayed Haji Ali
# Email: mh155@rice.edu
# Date: 8 June 2024

# based on code from 
# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import os
import argparse
import torch
import shutil
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from src.tools.logger import Logger
from pytorch_lightning import Trainer
from src.utilities.model.model_checkpoint import S3ModelCheckpoint

from src.utilities.data.videoaudio_dataset import VideoAudioDataset
from src.modules.latent_encoder.autoencoder_1d import AutoencoderKL1D
from src.tools.training_utils import get_restore_step
from src.tools.configuration import Configuration

def main(configs, config_yaml_path, exp_group_name, exp_name, debug=False):
    if "precision" in configs['training'].keys():
        torch.set_float32_matmul_precision(
            configs['training']["precision"]
        )  # highest, high, medium
        
    batch_size = configs["model"]["params"]["batchsize"]
    
    max_epochs = configs['step']['max_epochs']
    limit_val_batches = configs["step"].get("limit_val_batches", None)
    limit_train_batches = configs["step"].get("limit_train_batches", None)
    
    log_path = configs['logging']["log_directory"]
    save_top_k = configs["logging"].get("save_top_k", -1)
    save_checkpoint_every_n_steps = configs["logging"].get("save_checkpoint_every_n_steps", 5000)

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    augment_p = configs['data']['augment_p'] if 'augment_p' in configs['data'] else 0.0
    dataset = VideoAudioDataset(configs, split="train", add_ons=dataloader_add_ons, load_video=False, load_audio=True, sample_single_caption=True, augment_p=augment_p)
    
    print("[INFO] Using batch size of:", batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=configs['data'].get('num_workers', 32),
        pin_memory=True,
        shuffle=True,
    )

    print(
        "[INFO] The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    val_dataset = VideoAudioDataset(configs, split="test", add_ons=dataloader_add_ons, load_video=False, load_audio=True, sample_single_caption=True)

    val_loader = DataLoader(
        val_dataset,
        num_workers=configs['data'].get('num_workers', 32),
        batch_size=batch_size,
    )

    devices = torch.cuda.device_count()
    bs, base_lr = batch_size, configs["model"]["base_learning_rate"]
    learning_rate = base_lr


    model = AutoencoderKL1D(
        ddconfig=configs["model"]["params"]["ddconfig"],
        lossconfig=configs["model"]["params"]["lossconfig"],
        embed_dim=configs["model"]["params"]["embed_dim"],
        image_key=configs["model"]["params"]["image_key"],
        base_learning_rate=learning_rate,
        subband=configs["model"]["params"]["subband"],
        sampling_rate=configs["preprocessing"]["audio"]["sampling_rate"],
    )

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")
    
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

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)
    
    config_copy_dir = os.path.join(wandb_path, 'config')
    os.makedirs(config_copy_dir, exist_ok=True)
    shutil.copy(config_yaml_path, config_copy_dir)

    model.set_log_dir(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)

    if len(os.listdir(checkpoint_path)) > 0 and "resume_training" in configs['training'] and configs['training']["resume_training"]:
        print("[INFO] Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        print("[INFO] Resuming from step", n_step)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("[INFO] Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("[INFO] Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("[INFO] Training from scratch")
        resume_from_checkpoint = None

    
    wandb_logger = Logger(
        config=configs, 
        checkpoints_directory=wandb_path,
        run_name="%s/%s" % (exp_group_name, exp_name),
        offline=debug
    ).get_logger()
    
    nodes_count = configs['training']["nodes_count"]
    if nodes_count == -1:
        if "WORLD_SIZE" in os.environ:
            nodes_count = int(os.environ["WORLD_SIZE"]) // torch.cuda.device_count()
        else:
            nodes_count = 1
            
    print("[INFO] Training on devices", devices)
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        num_sanity_val_steps=1,
        num_nodes=nodes_count,
        limit_train_batches=limit_train_batches, 
        limit_val_batches=limit_val_batches,  
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=configs["model"]["params"].get("clip_grad", None)
    )

    # TRAINING
    trainer.fit(model, loader, val_loader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--autoencoder_config",
        type=str,
        required=True,
        help="path to autoencoder config .yam",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="debug mode",
    )

    args = parser.parse_args()

    config_yaml = args.autoencoder_config
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))
    
    configuration = Configuration(config_yaml)
    configs = configuration.get_config()

    main(configs, config_yaml, exp_group_name, exp_name, args.debug)
