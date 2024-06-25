# Author: Moayed Haji Ali
# Email: mh155@rice.edu
import os
import argparse
import yaml
from loguru import logger
from datetime import datetime
import numpy as np
from copy import deepcopy

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

from src.utilities.model.model_utils import setup_seed, set_logger, get_train_val_test_dataloaders
from src.tools.configuration import Configuration
from datetime import datetime



def main():
    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, type=str,
                        help='path to the pretrained checkpoint')
    parser.add_argument('-n', '--exp_name', default='', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', required=True, type=str, help='Name of the setting file.')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='Evaluation seed.')
    parser.add_argument('-d', '--debug', action="store_true",
                        help='debug mode.')
    parser.add_argument('-nw', '--num_workers', default=None, type=int,
                        help='number of workers for dataloader')
    
    args = parser.parse_args()

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

    config['training']["inference_exp_name"] = f"inference_{os.path.basename(args.config)[:-5]}" # remove .yaml

    exp_name = config['training']["inference_exp_name"]
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
    
    # data loading
    if args.checkpoint_path is not None:
        config['training']["pretrain_path"] = args.checkpoint_path
        
    assert config['training']["pretrain_path"] is not None, "Please provide a checkpoint"
    
    model = config['target'](config)
    model.eval()
    
    # data loading
    val_loaders, test_loaders = get_train_val_test_dataloaders(config, 
                                                                return_train=False,
                                                                return_val=True,
                                                                return_test=True,
                                                                augment_train=False, cache_dir=None) # Important: don't augment train

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Total numer of trainable parameters: {sum([i.numel() for i in model.parameters() if i.requires_grad])}')
    for val_k, val_loader in val_loaders.items():
        main_logger.info(f'Size of {val_k} validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    
    for test_k, test_loader in test_loaders.items():
        main_logger.info(f'Size of {test_k} test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    
    # update the model with data types, combine test and val loaders 
    val_loaders = {}
    val_loaders.update(test_loaders)
    print("[INFO] validation loaders", len(val_loaders), val_loaders.keys())
    model.val_loaders_labels = list(val_loaders.keys())
    val_loaders = list(val_loaders.values())
    model.log_output_dir = log_output_dir


    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_val_batches=config['training'].get('limit_val_batches', None),
    )

    os.makedirs(model_output_dir, exist_ok=True)
    with open(os.path.join(model_output_dir, 'config.yaml'), 'w+') as f:
        yaml.dump(config, f)

    
    # calculate metrics
    metrics = trainer.validate(model, dataloaders=val_loaders)
    print("[INFO] Done evaluation!")



if __name__ == '__main__':
    main()
