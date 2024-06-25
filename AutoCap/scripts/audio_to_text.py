#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import __init__
import argparse

import torch
from pytorch_lightning import Trainer, seed_everything

from src.utilities.model.model_utils import setup_seed, set_logger
from src.tools.configuration import Configuration
from src.tools.download_manager import get_checkpoint_path
from src.utilities.audio.audio_processing import read_wav_file



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
    parser.add_argument('-path', '--wav_path', default='',
                        required=True, type=str,
                        help='path to your wav file')
    
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, type=str,
                        help='path to the pretrained checkpoint')
    parser.add_argument('-c', '--config', default=None, type=str,
                        required=False,
                        help='Path to the setting file.')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='Inference seed.')
    
    parser.add_argument('-bs', '--beam_size', default=2, type=int,
                        help='beam size to use for captioning')
    parser.add_argument('--cpu', action="store_true",help='run on cpu only')
    parser.add_argument('--title', type=str, default="",
                        help='Title of your video')
    parser.add_argument('--video_caption', type=str, default="",
                        help='Accurate caption of the video')
    parser.add_argument('--description', type=str, default="",
                        help='Detailed description of the video')

    args = parser.parse_args()

    
    if args.config is None:
        args.config = get_checkpoint_path(f"{args.model}_config")
    if args.checkpoint_path is None:
        args.checkpoint_path = get_checkpoint_path(args.model)
        
        
    configuration = Configuration(args.config)
    config = configuration.get_config()
    
    if args.seed is not None:
        config['training']["seed"] = args.seed

    resampling_rate = config['data_args']['preprocessing']['audio']['sampling_rate']
    duration = config['data_args']['preprocessing']['audio']['duration']
    device = 'cpu' if args.cpu else 'cuda'
    
    # set up model
    setup_seed(config['training']["seed"])
    seed_everything(config['training']["seed"])

    # data loading
    if args.checkpoint_path is not None:
        config['training']["pretrain_path"] = args.checkpoint_path
        
    assert config['training']["pretrain_path"] is not None, "Please provide a checkpoint"


    model = config['target'](config).to(device)
    model.eval()

    wav, _ = read_wav_file(args.wav_path,
                            resampling_rate=resampling_rate,
                            duration=duration)
    
    wav = torch.Tensor(wav).to(device)
    
    meta = model.get_meta_dict({
        "title":[args.title],
        "video_caption": [args.video_caption],
        "description": [args.description]
    })
    caption = model.generate(samples=wav,
                    meta=meta,
                    num_beams=args.beam_size)

    print(caption)
             



if __name__ == '__main__':
    main()
