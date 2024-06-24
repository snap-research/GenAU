"""For a given parent directory, it consideres all of their subdirectories as different experiments. For each experiment, it finds all subdirectories that start with "val" and compute the metrics on this subdirectory. 
The subdirectory should contain wav files with the same name as the test dataset directory.
"""

import os
import torch
import shutil
from audioldm_eval import EvaluationHelper
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from src.utilities.data.videoaudio_dataset import VideoAudioDataset
from src.tools.training_utils import (
    get_restore_step,
    copy_test_subset_data,
)
from src.utilities.model.model_util import instantiate_from_config
from src.tools.configuration import Configuration



SAMPLE_RATE = 16000
devices = torch.cuda.device_count()
evaluator = EvaluationHelper(SAMPLE_RATE, torch.device(f"cuda:{0}"))


def locate_yaml_file(path):
    for file in os.listdir(path):
        if ".yaml" in file:
            return os.path.join(path, file)
    return None


def is_evaluated(path):
    candidates = []
    for file in os.listdir(
        os.path.dirname(path)
    ):  # all the file inside a experiment folder
        if ".json" in file:
            candidates.append(file)
    folder_name = os.path.basename(path)
    for candidate in candidates:
        if folder_name in candidate:
            return True
    return False


def locate_validation_output(path):
    folders = []
    for file in os.listdir(path):
        dirname = os.path.join(path, file)
        if "val_" in file and os.path.isdir(dirname):
            if not is_evaluated(dirname):
                folders.append(dirname)
    return folders


def evaluate_exp_performance(exp_name):
    abs_path_exp = os.path.join(latent_diffusion_model_log_path, exp_name)
    config_yaml_path = locate_yaml_file(abs_path_exp)

    if config_yaml_path is None:
        print("%s does not contain a yaml configuration file" % exp_name)
        return

    folders_todo = locate_validation_output(abs_path_exp)

    for folder in folders_todo:
        if len(os.listdir(folder)) == 964:
            test_dataset = "audiocaps"
        elif len(os.listdir(folder)) > 5000:
            test_dataset = "musiccaps"
        else:
            print(f"[Warning, generate_and_eval.py] cannot identiy test dataset name at folder {folder}")
            print(f"[INFO] Skipping folder {folder}")
            continue

        test_audio_data_folder = os.path.join(test_audio_path, test_dataset)

        evaluator.main(folder, test_audio_data_folder)

@torch.no_grad()
def generate_test_audio(configs, config_yaml_path, exp_group_name, exp_name, use_wav_cond=False, strategy='wo_ema', batch_size=244, n_candidates_per_samples=1, ckpt=None):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs['training'].keys():
        torch.set_float32_matmul_precision(
            configs['training']["precision"]
        )  # highest, high, medium


    log_path = configs['logging']["log_directory"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    batch_size = configs["model"]["params"]["batchsize"]

    # set up evaluation parameters
    configs['model']['params']['evaluation_params']['n_candidates_per_samples'] = n_candidates_per_samples
    if ckpt is not None:
        configs["reload_from_ckpt"] = ckpt

    val_dataset = VideoAudioDataset(configs, split="test", add_ons=dataloader_add_ons, load_video=False, load_audio=True, sample_single_caption=True)

    val_loader = DataLoader(
        val_dataset,
        num_workers=12, # configs['data'].get('num_workers', 12),
        batch_size=max(1, batch_size // configs['model']['params']['evaluation_params']['n_candidates_per_samples']),
    )

    config_reload_from_ckpt = configs.get("reload_from_ckpt", None)

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)

    if config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        try:
            n_step = int(resume_from_checkpoint.split(".ckpt")[0].split("step=")[1]) 
        except:
            print("[Warning] cannot extract model step from the checkpoint filename, using UNK")
            n_step = "UNK"
            
        print("Reload given checkpoint %s" % resume_from_checkpoint)
    elif len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    else:
        raise "Please specify a pre-trained checkpoint"

    guidance_scale = configs["model"]["params"]["evaluation_params"][
        "unconditional_guidance_scale"
    ]
    ddim_sampling_steps = configs["model"]["params"]["evaluation_params"][
        "ddim_sampling_steps"
    ]
    n_candidates_per_samples = configs["model"]["params"]["evaluation_params"][
        "n_candidates_per_samples"
    ]
    configs['model']['params']['ckpt_path'] = resume_from_checkpoint

    
    # change log directory
    configs['logging']["log_directory"] = configs['logging']["log_directory"].replace('train', 'evaluation')
    
    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(configs['logging']["log_directory"], exp_group_name, exp_name)
    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()
    
    if use_wav_cond:
        latent_diffusion.random_clap_condition(text_prop=0.0)
    
    name = latent_diffusion.get_validation_folder_name(guidance_scale, ddim_sampling_steps, n_candidates_per_samples, step=n_step, tag=strategy)
    if strategy == 'wo_ema':
        print("[INFO] Using No EMA strategy")
        latent_diffusion.use_ema = False
    
    latent_diffusion.name = name
    latent_diffusion.unconditional_guidance_scale = guidance_scale
    latent_diffusion.ddim_sampling_steps = ddim_sampling_steps
    latent_diffusion.n_gen = n_candidates_per_samples
    latent_diffusion.generate_sample(
        val_loader,
        name=name,
        unconditional_guidance_scale=guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_gen=n_candidates_per_samples,
        use_ema=(strategy != 'wo_ema')
    )
    
    
    # copy test data if it does not exists
    test_data_subset_folder = os.path.join(
        os.path.dirname(configs['logging']["log_directory"]),
        "testset_data",
        val_dataset.dataset_name,
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    copy_test_subset_data(val_dataset, test_data_subset_folder)
    

def eval(exps):
    for exp in exps:
        try:
            evaluate_exp_performance(exp)
        except Exception as e:
            print(exp, e)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="AudioLDM model evaluation")
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=True,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        required=False,
        default='wo_ema',
        help="The strategy of combining weights from different checkpoint: wo_ema, avg_ckpt, or ema",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=64
    )
    
    parser.add_argument(
        "-nc",
        "--n_candidates_per_samples",
        type=int,
        required=False,
        default=1,
        help="Normally set it to 1, "
    )
    
    parser.add_argument(
        "-ckpt",
        type=str,
        default=None
    )

    args = parser.parse_args()
    assert args.strategy in ['wo_ema', 'avg_ckpt', 'ema']

    config_yaml = args.config_yaml
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    configuration = Configuration(config_yaml_path)
    configs = configuration.get_config()

    # generate audio
    generate_test_audio(configs, config_yaml_path, exp_group_name, exp_name, strategy=args.strategy, batch_size=args.batch_size, n_candidates_per_samples=args.n_candidates_per_samples, ckpt=args.ckpt)
    
    test_audio_path = os.path.join(
        os.path.dirname(configs['logging']["log_directory"]),
        "testset_data"
    )
    latent_diffusion_model_log_path = os.path.join(configs['logging']["log_directory"], exp_group_name)
    
    # copy config path
    shutil.copy(config_yaml_path, os.path.join(configs['logging']["log_directory"], exp_group_name, exp_name))
    
    eval([exp_name])