import __init__
import os
import argparse
import yaml
from loguru import logger
from datetime import datetime
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
import torchaudio

from src.utilities.model.model_utils import setup_seed, set_logger, decode_output
from src.tools.configuration import Configuration
from src.tools.download_manager import get_checkpoint_path
from src.utilities.audio.audio_processing import read_wav_file
from src.tools.io import load_yaml_file



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
    parser.add_argument('-f', '--folder_path', default='audio_samples/ood_samples', type=str,
                        required=True,
                        help='path to folder containing the audios.')
    parser.add_argument('-n', '--exp_name', default='', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='Name of the setting file.')
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, type=str,
                        help='path to the pretrained checkpoint')
    parser.add_argument('-meta', '--meta_data_file', default=None, type=str,
                        help='Path to the meta data yaml file.')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='Inference seed.')
    parser.add_argument('--cpu', action="store_true",
                        help='run on cpu only')
    parser.add_argument('-bs', '--beam_size', default=2, type=int,
                        help='beam size to use for captioning')

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

    resampling_rate = config['data_args']['preprocessing']['audio']['sampling_rate']
    duration = config['data_args']['preprocessing']['audio']['duration']
    device = 'cpu' if args.cpu else 'cuda'

    setup_seed(config['training']["seed"])
    seed_everything(config['training']["seed"])

    config['training']["inference_exp_name"] = f"samples_{os.path.basename(args.config)[:-5]}" # remove .yaml

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

    if args.checkpoint_path is not None:
        config['training']["pretrain_path"] = args.checkpoint_path
        
    assert config['training']["pretrain_path"] is not None, "Please provide a checkpoint"


    model = config['target'](config).to(device)
    model.eval()

    # read meta data if provided
    meta_data_dict = {}
    if args.meta_data_file is not None:
        meta_data_dict = load_yaml_file(args.meta_data_file)
    
    # caption training
    wav_list = []
    meta_list = []
    fnames_list = []
    for filepath in os.listdir(args.folder_path):
        if not filepath.endswith('.wav'):
            continue
        try: 
            wav, _ = read_wav_file(os.path.join(args.folder_path, filepath),
                                   resampling_rate=resampling_rate,
                                   duration=duration)
            wav_list.append(torch.Tensor(wav).to(device))
            
            meta_list.append(model.get_meta_dict({"title":[meta_data_dict.get(filepath, {}).get('title', '')],
                                    "video_caption": [meta_data_dict.get(filepath, {}).get('video_caption', '')],
                                    "description": [meta_data_dict.get(filepath, {}).get('description', '')]})[0])
            fnames_list.append(filepath)
            
        except Exception as e:
            print(f"[ERROR] Error processing {os.path.join(args.folder_path, filepath)}:", e)
    
    captions = model.generate(samples= torch.cat(wav_list),
                            meta=meta_list,
                            num_beams=args.beam_size)

    
    # log captions
    logging_path = str(log_output_dir) + '/beam_captions_{}bsize.txt'.format(args.beam_size)
    logging = logger.add(logging_path,
                            format='{message}', level='INFO',
                            filter=lambda record: record['extra']['indent'] == 3)
    caption_logger = logger.bind(indent=3)
    captions_pred, f_names = [], []

    for pred_cap, f_name in zip(captions, fnames_list):
        f_names.append(f_name)
        captions_pred.append({'file_name': f_name, 'caption_predicted': pred_cap})

        log_strings = [f'{f_name}: {pred_cap}']
        

        [caption_logger.info(log_string)
         for log_string in log_strings]
    logger.remove(logging)
    print(f"[INFO] captions are saved at {logging_path}")
             



if __name__ == '__main__':
    main()
