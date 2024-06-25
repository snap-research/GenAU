#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import __init__
import os
import argparse
import yaml
from loguru import logger
from datetime import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

from src.utilities.model.model_utils import setup_seed, set_logger, get_train_val_test_dataloaders
from src.tools.configuration import Configuration
from src.tools.download_manager import get_checkpoint_path

def main():
    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='autocap-full',
        required=False,
        help="model name",
    )
    parser.add_argument('-n', '--exp_name', default='', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-k', '--caption_store_key', required=True, type=str, help='Name of the experiment.')
    parser.add_argument('-c', '--config', required=False, type=str, help='Name of the setting file.')
    parser.add_argument('-bs', '--beam_size', required=True, type=int,
                        help='beam size to use for captioning')
    parser.add_argument('-l', '--lr', default=None, type=float,
                        help='Learning rate.') 
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='Training seed.')
    parser.add_argument('-d', '--debug', default=False, type=bool,
                        help='debug mode.')
    parser.add_argument('-nw', '--num_workers', default=None, type=int,
                        help='number of workers for dataloader')
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, type=str,
                        help='path to the pretrained checkpoint')
    parser.add_argument('--start_idx', '-st', default=0, type=int,
                        help='start index for data loader')

    parser.add_argument('--end_idx', '-e', default=1000000000, type=int,
                        help='end index for data loader')
    
    parser.add_argument(
        "--dataset_keys",
        nargs='*',
        required=False,
        default=[],
        help="A list of dataset keys for training or finetuning",
    )
    
    args = parser.parse_args()
    

    if args.config is None:
        args.config = get_checkpoint_path(f"{args.model}_config")
    if args.checkpoint_path is None:
        args.checkpoint_path = get_checkpoint_path(args.model)
        
    configuration = Configuration(args.config)
    config = configuration.get_config()

    if args.exp_name:
        config['training']["exp_name"] = args.exp_name
    
    if args.seed is not None:
        config['training']["seed"] = args.seed

    if args.num_workers is not None: 
        config['data_args']['num_workers']
    

    # set up model
    devices = torch.cuda.device_count()
    setup_seed(config['training']["seed"])
    seed_everything(config['training']["seed"])

    config["inference_exp_name"] = f"inference_{os.path.basename(args.config)[:-5]}" # remove .yaml
    config['beam_size'] = args.beam_size
    config['caption_key'] = args.caption_store_key

    exp_name = config["inference_exp_name"]
    parent_log_dir = os.path.join("run_logs", 'inference', exp_name)
    os.makedirs(parent_log_dir, exist_ok=True)

    folder_name = '{}/lr_{}_batch_{}_seed_{}_date_{}'.format(exp_name,
                                                     config["optim_args"]["lr"],
                                                     config["data_args"]["batch_size"],
                                                     config['training']["seed"],
                                                     datetime.now().strftime("%Y-%m-%d-%H:%M"),)
    

    model_output_dir, log_output_dir = set_logger(parent_log_dir, folder_name)
    main_logger = logger.bind(indent=1)

    # save a copy of config
    os.makedirs(model_output_dir, exist_ok=True)
    with open(os.path.join(model_output_dir, 'config.yaml'), 'w+') as f:
        yaml.dump(config, f)

    main_logger.info(f'Process on devices: {devices}')

    if args.dataset_keys:
        config['data_args']['data']['train'] = args.dataset_keys
        
    # checkpoint 
    if args.checkpoint_path is not None:
        config['training']["pretrain_path"] = args.checkpoint_path
    
    # model
    model = config['target'](config)
    
    # data loading
    train_loader  = get_train_val_test_dataloaders(config, 
                                                    return_train=True,
                                                    return_val=False,
                                                    return_test=False,
                                                    augment_train=False, cache_dir=None) # Important: don't augment train and do not use a cache dir

    model.train_loader_len = len(train_loader) // devices

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Total numer of trainable parameters: {sum([i.numel() for i in model.parameters() if i.requires_grad])}')
    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    # for val_k, val_loader in val_loaders.items():
    #     main_logger.info(f'Size of {val_k} validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    
    # for test_k, test_loader in test_loaders.items():
    #     main_logger.info(f'Size of {test_k} test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    
    model.log_output_dir = log_output_dir
    model.caption_store_key = args.caption_store_key


    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_val_batches=config['training'].get('limit_val_batches', None),
    )
    
    
    # trainer
    model.eval()
    
    # caption training
    model.eval_beam_sizes = [args.beam_size]

    trainer.predict(model, dataloaders=train_loader)

    # for val_k, val_loaders in val_loaders.items():
    #     trainer.predict(model, dataloaders=val_loader)
    
    # for test_k, test_loader in test_loaders.items():
    #     trainer.predict(model, dataloaders=test_loader) 



if __name__ == '__main__':
    main()
