import os
import wget
from pathlib import Path

save_dir = "pretrained_models"

checkpoint_paths = {"HTSAT": "pretrained_models/audio_encoder/HTSAT.ckpt",
                    "music_speech_audioset_epoch_15_esc_89.98": "pretrained_models/clap/music_speech_audioset_epoch_15_esc_89.98.pt.pt",
                    "clap_htsat_tiny": "pretrained_models/clap/clap_htsat_tiny.pt",
                    "hifigan_16k_64bins": "pretrained_models/vocoder/hifigan_16k_64bins.ckpt",
                    "hifigan_16k_64bins_config": "pretrained_models/vocoder/hifigan_16k_64bins.json",
                    "1dvae_64ch_16k_64bins": "pretrained_models/vae/1dvae_64ch_16k_64bins.ckpt",
                    "genau-full-l": "pretrained_models/genau/genau-full-l.ckpt",
                    "genau-full-l_config": "pretrained_models/genau/genau-full-l.yaml",
                    "genau-full-s": "pretrained_models/genau/genau-full-s.ckpt",
                    "genau-full-s_config": "pretrained_models/genau/genau-full-s.yaml"} 


checkpoint_urls = {"HTSAT":'https://drive.usercontent.google.com/download?id=11XiCDsW3nYJ6uM87pvP3wI3pDAGhsBC1&export=download&confirm=t&uuid=986f3e02-6fc4-4419-ab91-ffb4017b2aba',
                   "music_speech_audioset_epoch_15_esc_89.98": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt",
                                      
                   "clap_htsat_tiny": "https://huggingface.co/mali6/autocap/resolve/main/clap_htsat_tiny.pt",
                    "hifigan_16k_64bins": "https://huggingface.co/mali6/autocap/resolve/main/hifigan_16k_64bins.ckpt",
                    "hifigan_16k_64bins_config": "https://huggingface.co/mali6/autocap/resolve/main/hifigan_16k_64bins.json",
                    "1dvae_64ch_16k_64bins": "https://huggingface.co/mali6/autocap/resolve/main/1dvae_64_344999.ckpt",
                    "genau-full-l": "https://huggingface.co/mali6/autocap/resolve/main/genau-full-l.ckpt",
                    "genau-full-l_config": "https://huggingface.co/mali6/autocap/resolve/main/genau-full-l.yaml",
                    "genau-full-s": "https://huggingface.co/mali6/autocap/resolve/main/genau-full-s.ckpt",
                    "genau-full-s_config": "https://huggingface.co/mali6/autocap/resolve/main/genau-full-s.yaml",}


def get_checkpoint_path(model_name, local_ckpt_path=None, download=True):
    if local_ckpt_path is None:
        assert model_name in checkpoint_paths.keys(), f"Cannot recognize model {model_name}"
        local_ckpt_path = checkpoint_paths[model_name]
    if os.path.exists(local_ckpt_path):
        return local_ckpt_path
    elif not download:
        raise f"[ERROR] model does not exist at {local_ckpt_path}, please use the download flag to attempt to downloaded it from the web or manually download the checkpoint and place at {local_ckpt_path}"
    else:
        Path(local_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        
        assert model_name in checkpoint_paths, f"[ERROR] Model {model_name} is not recognized and the pretrained checkpoint does not exist at {local_ckpt_path}.\n Available models to the download manager are {checkpoint_paths.keys()}"
        
        print(f"[INFO] downloading model {model_name} at {local_ckpt_path}")
        wget.download(checkpoint_urls[model_name], local_ckpt_path)
        print(f"[INFO] Checkpoint for model {model_name} is downloaded at {local_ckpt_path}")
        return local_ckpt_path
        