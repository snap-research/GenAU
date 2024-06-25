#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Evaluation tools adapted from https://github.com/fartashf/vsepp/blob/master/evaluation.py
"""

import numpy as np
import random
import sys
from loguru import logger
from pathlib import Path
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from src.utilities.data.videoaudio_dataset import VideoAudioDataset, custom_collate_fn

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_logger(parent_dir, exp_name):
    log_output_dir = Path(parent_dir, exp_name, 'logging')
    model_output_dir = Path(parent_dir, exp_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    return model_output_dir, log_output_dir


def decode_output(predicted_output, ref_captions, file_names,
                  log_output_dir, epoch, beam_size=1):

    if beam_size != 1:
        logging = logger.add(str(log_output_dir) + '/beam_captions_{}ep_{}bsize.txt'.format(epoch, beam_size),
                             format='{message}', level='INFO',
                             filter=lambda record: record['extra']['indent'] == 3)
        caption_logger = logger.bind(indent=3)
        caption_logger.info('Captions start')
        caption_logger.info('Beam search:')
    else:
        logging = logger.add(str(log_output_dir) + '/captions_{}ep.txt'.format(epoch),
                             format='{message}', level='INFO',
                             filter=lambda record: record['extra']['indent'] == 2)
        caption_logger = logger.bind(indent=2)
        caption_logger.info('Captions start')
        caption_logger.info('Greedy search:')

    captions_pred, captions_gt, f_names = [], [], []

    for pred_cap, gt_caps, f_name in zip(predicted_output, ref_captions, file_names):
        f_names.append(f_name)
        captions_pred.append({'file_name': f_name, 'caption_predicted': pred_cap})
        ref_caps_dict = {'file_name': f_name}
        for i, cap in enumerate(gt_caps):
            ref_caps_dict[f"caption_{i + 1}"] = cap
        captions_gt.append(ref_caps_dict)

        log_strings = [f'Captions for file {f_name}:',
                       f'\t Predicted caption: {pred_cap}']
        
        for idx, c in enumerate(gt_caps):
            log_strings.append(f'\t Original caption_{idx}: {c}')

        [caption_logger.info(log_string)
         for log_string in log_strings]
    logger.remove(logging)
    return captions_pred, captions_gt

def get_train_val_test_dataloaders(config, return_train=True, return_val=True, return_test=True, augment_train=True, 
                                   cache_dir=None, dataset_kwargs={}, **kwargs):
    ret_dataloaders = []

    if return_train:
        augment_p = 0.0
        if augment_train:
            augment_p = config['data_args']['augmentation_p']
        dataset_config = deepcopy(config['data_args'])
        
        train_dataset = VideoAudioDataset(config=dataset_config, split='train',
                                waveform_only=True, load_video=False, keep_audio_files=True,
                                sample_single_caption=True, augment_p=augment_p, cache_dir=cache_dir, **dataset_kwargs)
        
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=config['data_args']['batch_size'],
                num_workers=config['data_args']['num_workers'],
                pin_memory=True,
                shuffle=True,
                drop_last=False,
                collate_fn=custom_collate_fn,
                **kwargs
            )
        ret_dataloaders.append(train_loader)
    

    if return_val:
        val_datasets = config['data_args']['data']['val'].copy()
        val_loaders = {}
        for val_k in val_datasets:
            dataset_config = deepcopy(config['data_args'])
            dataset_config['data']['val'] = val_k
            val_dataset = VideoAudioDataset(config=dataset_config, split='val',
                                    waveform_only=True, load_video=False, keep_audio_files=True,
                                    sample_single_caption=False, cache_dir=cache_dir,  **dataset_kwargs)
            

            val_loaders[f'val/{val_k}'] = DataLoader(
                    dataset=val_dataset,
                    batch_size=config['data_args']['batch_size'],
                    num_workers=config['data_args']['num_workers'],
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=custom_collate_fn,
                    **kwargs
                )
        ret_dataloaders.append(val_loaders)
    
    if return_test:
        test_datasets = config['data_args']['data']['test'].copy()
        test_loaders = {}

        for test_k in test_datasets:
            dataset_config = deepcopy(config['data_args'])
            dataset_config['data']['test'] = test_k
            test_dataset = VideoAudioDataset(config=dataset_config, split='test',
                                    waveform_only=True, load_video=False, keep_audio_files=True,
                                    sample_single_caption=False, cache_dir=cache_dir, **dataset_kwargs)
            

            test_loaders[f'test/{test_k}'] = DataLoader(
                    dataset=test_dataset,
                    batch_size=config['data_args']['batch_size'],
                    num_workers=config['data_args']['num_workers'],
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=custom_collate_fn,
                    **kwargs
                )
        ret_dataloaders.append(test_loaders)
    if len(ret_dataloaders) == 1:
        return ret_dataloaders[0]
    return ret_dataloaders