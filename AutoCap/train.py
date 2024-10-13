# Author: Moayed Haji Ali
# Email: mh155@rice.edu


import os
import argparse
from pprint import PrettyPrinter
import yaml
from loguru import logger
from datetime import datetime
from copy import deepcopy

import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer, seed_everything

from src.utilities.model.model_utils import setup_seed, set_logger, get_train_val_test_dataloaders
from src.utilities.model.model_checkpoint import S3ModelCheckpoint
from src.tools.configuration import Configuration
from src.tools.logger import Logger


def main():
    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', default='settings/audio_qformer_audiocaps_settings.yaml', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-l', '--lr', default=None, type=float,
                        help='Learning rate.') 
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='Training seed.')
    parser.add_argument('-d', '--debug', action="store_true", help='debug mode.')
    parser.add_argument('-nw', '--num_workers', default=None, type=int,
                        help='number of workers for dataloader')
    
    args = parser.parse_args()

    configuration = Configuration(args.config)
    config = configuration.get_config()

    if args.exp_name:
        config["training"]["exp_name"] = args.exp_name
    
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    
    if args.lr is not None:
        config["optim_args"]["lr"] = args.lr

    if args.num_workers is not None: 
        config['data_args']['num_workers'] = args.num_workers
    
    # set up model
    devices = torch.cuda.device_count()
    config['training']['devices'] = devices

    print("seed", config["training"]["seed"])
    setup_seed(config["training"]["seed"])
    seed_everything(config["training"]["seed"])

    if "exp_name" not in config['training'] or not config['training']["exp_name"]:
        config['training']["exp_name"] = os.path.basename(args.config)[:-5] # remove .yaml

    exp_name = config['training']["exp_name"]
    parent_log_dir = os.path.join("run_logs", 'train', exp_name)
    os.makedirs(parent_log_dir, exist_ok=True)
    print("seed", config["training"]["seed"])
    folder_name = '{}/lr_{}_batch_{}_seed_{}_date_{}'.format(exp_name,
                                                     config["optim_args"]["lr"],
                                                     config["data_args"]["batch_size"],
                                                     config["training"]["seed"],
                                                     datetime.now().strftime("%Y-%m-%d-%H:%M"),)

    model_output_dir, log_output_dir = set_logger(parent_log_dir, folder_name)
    main_logger = logger.bind(indent=1)

    # save a copy of config
    os.makedirs(model_output_dir, exist_ok=True)
    with open(os.path.join(model_output_dir, 'config.yaml'), 'w+') as f:
        yaml.dump(config, f)

    main_logger.info(f'Process on devices: {devices}')
    model = config['target'](config)

    # data loading
    train_loader, val_loaders = get_train_val_test_dataloaders(config,
                                                                return_train=True,
                                                                return_val=True,
                                                                return_test=False,
                                                                cache_dir=None)
    
    
    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')
    

    model.train_loader_len = len(train_loader) // devices

    wandb_logger = Logger(
        config=config,
        run_name=folder_name,
        checkpoints_directory=model_output_dir,
        offline=args.debug
    ).get_logger()

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Total numer of trainable parameters: {sum([i.numel() for i in model.parameters() if i.requires_grad])}')
    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    for val_k, val_loader in val_loaders.items():
        main_logger.info(f'Size of {val_k} validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')

    
    
    print("val_loaders", len(val_loaders), val_loaders.keys())
    model.val_loaders_labels = list(val_loaders.keys())
    val_loaders = list(val_loaders.values())
    model.log_output_dir = log_output_dir


    # ckpt
    validation_every_n_epochs = config["step"].get("validation_every_n_epochs", None)
    save_checkpoint_every_n_epochs = config["logging"]["save_checkpoint_every_n_epochs"]
    
    checkpoint_callback = S3ModelCheckpoint(
        bucket_name=config['logging'].get('S3_BUCKET', None),
        s3_folder=config['logging'].get('S3_FOLDER', None),
        dirpath=model_output_dir,
        filename="checkpoint-epoch={epoch}-global_step={global_step:.0f}",
        every_n_epochs=save_checkpoint_every_n_epochs,
        save_top_k=config['logging'].get('save_top_k', -1),
        save_last=True,
    )

    # training
    print("[INFO]==> Save checkpoint every %s steps" % save_checkpoint_every_n_epochs)
    print("[INFO]==> Perform validation every %s epochs" % validation_every_n_epochs)

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        max_epochs=config["step"]["epochs"],
        num_sanity_val_steps=config['step'].get('num_sanity_val_steps', 1),
        limit_val_batches=config['step'].get('limit_val_batches', None),
        limit_train_batches=config['step'].get('limit_train_batches', None),
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        gradient_clip_val=config["model"].get("clip_grad", None),
        profiler=config['training'].get('profiler', None),
    )


    resume_ckpt = config['training'].get("pretrain_path", None)
    if config['training']["pretrain"] and resume_ckpt is not None:
        pretrain_checkpoint = torch.load(config['training']["pretrain_path"])
        model.load_state_dict(pretrain_checkpoint['state_dict'])
        main_logger.info(f"Loaded weights from {config['training']['pretrain_path']} for pretraining and starting finetuning for scratch")
        print("weights loaded", flush=True)
        trainer.fit(model, train_loader, val_loaders)
    else:
        if resume_ckpt is not None:
            main_logger.info(f"Resume training from {config['training']['pretrain_path']}")
        trainer.fit(model, train_loader, val_loaders, ckpt_path=resume_ckpt)

    # Training done, evaluate on evaluation set
    main_logger.info('Training done.')


if __name__ == '__main__':
    main()
