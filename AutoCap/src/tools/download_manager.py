import os
import wget
from pathlib import Path

save_dir = "pretrained_models"

checkpoint_paths = {"HTSAT": "pretrained_models/audio_encoder/HTSAT.ckpt",
                    "music_speech_audioset_epoch_15_esc_89.98": "pretrained_models/clap/music_speech_audioset_epoch_15_esc_89.98.pt.pt",
                    "autocap-full": "pretrained_models/autocap/autocap-full.ckpt",
                    "autocap-full_config": "pretrained_models/autocap/autocap-full.yaml"} 

checkpoint_urls = {"HTSAT":'https://drive.usercontent.google.com/download?id=11XiCDsW3nYJ6uM87pvP3wI3pDAGhsBC1&export=download&confirm=t&uuid=986f3e02-6fc4-4419-ab91-ffb4017b2aba',
                   "music_speech_audioset_epoch_15_esc_89.98": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt",
                   "autocap-full": "https://huggingface.co/mali6/autocap/resolve/main/autocap-full.ckpt",
                   "autocap-full_config": "https://huggingface.co/mali6/autocap/resolve/main/autocap-full.yaml",
                   
                   # AutoCAP pretrained checkpoints
                   "autocap_pretraining": "TODO",
                   "autocap_audiocaps": "TODO",}


def get_checkpoint_path(model_name, local_ckpt_path=None, download=True):
    if local_ckpt_path is None:
        local_ckpt_path = checkpoint_paths[model_name]
    if os.path.exists(local_ckpt_path):
        return local_ckpt_path
    elif not download:
        raise f"[ERROR] model does not exist at {local_ckpt_path}, please use the download flag to attempt to downloaded it from the web or manually download the checkpoint and place at {local_ckpt_path}"
    else:
        Path(local_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        
        if model_name not in checkpoint_paths:
            raise f"[ERROR] Model {model_name} is not recognized and the pretrained checkpoint does not exist at {local_ckpt_path}.\n Available models to the download manager are {checkpoint_paths.keys()}"
        wget.download(checkpoint_urls[model_name], local_ckpt_path)
        print(f"[INFO] Checkpoint for model {model_name} is downloaded at {local_ckpt_path}")
        return local_ckpt_path
        