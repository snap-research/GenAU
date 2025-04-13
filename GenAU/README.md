
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://snap-research.github.io/GenAU) [![Arxiv](https://img.shields.io/badge/arxiv-2406.19388-b31b1b)](https://arxiv.org/abs/2406.19388) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/taming-data-and-transformers-for-audio-1/audio-captioning-on-audiocaps)](https://paperswithcode.com/sota/audio-captioning-on-audiocaps?p=taming-data-and-transformers-for-audio-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/taming-data-and-transformers-for-audio-1/audio-generation-on-audiocaps)](https://paperswithcode.com/sota/audio-generation-on-audiocaps?p=taming-data-and-transformers-for-audio-1)

# GenAU inference, training and evaluation
- [Introduction](#introduction)
- [Environemnt setup](#environment-initalization)
- [Inference](#inference)
    * [Audio to text script](#text-to-audio)<!-- * [Gradio demo](#gradio-demo) -->
    * [Inference a list of promots](#inference-a-list-of-prompts)
- [Training](#training)
    * [GenAU](#genau)
    * [Finetuning GenAU](#finetuning-genau)
    * [1D-VAE (optional)](#1d-vae-optional)
- [Evaluation](#evaluation)
- [Cite this work](#cite-this-work)
- [Acknowledgements](#acknowledgements)

# Introduction 
We introduce GenAU, a transformer-based audio latent diffusion model leveraging the FIT architecture. Our model compresses mel-spectrogram data into a 1D representation and utilizes layered attention processes to achieve state-of-the-art audio generation results among open-source models.
<br/>

<div align="center">
<img src="../assets/genau.png" width="900" />
</div>

<br/>

# Environment initalization
For initializing your environment, please refer to the [general README](../README.md).

# Inference

## Text to Audio
To quickly generate an audio based on an input text prompt, run
```shell
python scripts/text_to_audio.py --prompt "Horses growl and clop hooves." --model "genau-full-l"
```
- This will automatically downloads and uses the model `genau-full-l` with default settings. You may change these parameters or provide your custome model config file and checkpoint path.
- Available models:
    - `genau-l-full-hq-data` (1.25B parameters) trained with AutoRecap-XL filtered with CLAP score of 0.4 (20.7M samples)
    - `genau-full-l` (1.25B parameters) trained with AutoRecap (760k samples)
    - `genau-full-s` (493M parameters) trained with AutoRecap (760k samples)
- These models are trained to generate ambient sounds and is incapable of generating speech or music.
- Outputs will be saved by default at `samples/model_output` using the provided prompt as the file name.

## Gradio Demo
Run a local interactive demo with Gradio:
```shell
python scripts/gradio_demo.py
```

## Inference a list of prompts
Optionally, you may prepare a `.txt` file with your target prompts and run

```shell
python scripts/inference_file.py --list_inference <path-to-prompts-file> --model <model_name>

# Example 
python scripts/inference_file.py --list_inference samples/prompts_list.txt --model "genau-full-l"
```


## Training

### Dataset
Please refer to the [dataset preperation README](../dataset_preperation/README.md) for instructions on downloading our dataset or preparing your own.

### GenAU
- Preapre a yaml config file for your experiments. A sample config file is provided at `settings/simple_runs/genau.yaml`
- Specify your project name and provide your Wandb key in the config file. A Wandb key can be obtained from [https://wandb.ai/authorize](https://wandb.ai/authorize)
- Optionally, provide your S3 bucket and folder to save intermediate checkpoints. 
- By default, checkpoints will be saved under `run_logs/genau/train` at the same level as the config file.

```shell
# Training GenAU from scratch
python train/genau.py -c settings/simple_runs/genau.yaml
```

For multinode training, run 
```shell
python -m torch.distributed.run --nproc_per_node=8 train/genau.py -c settings/simple_runs/genau.yaml
```
### Finetuning GenAU

- Prepare you custom dataset and obtain the dataset keys following [dataset preperation README](../dataset_preperation/README.md) 
- Make a copy and adjust the default config file of `genau-full-l` which you can find under `pretrained_models/genau/genau-full-l.yaml`
- Add ids for your dataset keys under `dataset2id` attribute in the config file.

```shell
# Finetuning GenAU 
python train/genau.py --reload_from_ckpt 'genau-full-l' \
                      --config <path-to-config-file> \
                      --dataset_keys "<dataset_key_1>" "<dataset_key_2>" ...
```


### 1D VAE (Optional)
By default, we offer a pre-trained 1D-VAE for GenAU training. If you prefer, you can train your own VAE by following the provided instructions.
- Prepare your own dataset following the instructions in the [dataset preperation README](../dataset_preperation/README.md) 
- Preapre your yaml config file in a similar way to the GenAU config file
- A sample config file is provided at `settings/simple_runs/1d_vae.yaml`

```shell
python train/1d_vae.py -c settings/simple_runs/1d_vae.yaml
```

## Evaluation
- We follow [audioldm](https://github.com/haoheliu/AudioLDM-training-finetuning) to perform our evaulations. 
- By default, the models will be evaluated periodically during training as specified in the config file. For each evaulation, a folder with the generated audio will be saved under `run_logs/train' at the same levels the specified config file. 
- The code idenfities the test dataset in an already existing folder according to that number of samples. If you would like to test on a new test dataset, register it in `scripts/generate_and_eval` or provide `--evaluation_dataset` name.

```shell

# Evaluate on an existing generated folder
python scripts/evaluate.py --log_path <path-to-the-experiment-folder>

# Geneate test audios from a pre-trained checkpoint and run evaulation
python scripts/generate_and_eval.py -c <path-to-config> -ckpt <path-to-pretrained-ckpt> --generate_and_eval audiocaps
```
The evaluation result will be saved in a json file at the same level of the generated audio folder.

# Cite this work
If you found this useful, please consider citing our work

```
@misc{hajiali2024tamingdatatransformersaudio,
      title={Taming Data and Transformers for Audio Generation}, 
      author={Moayed Haji-Ali and Willi Menapace and Aliaksandr Siarohin and Guha Balakrishnan and Vicente Ordonez},
      year={2024},
      eprint={2406.19388},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2406.19388}, 
}
```

# Acknowledgements
Our audio generation and evaluation codebase relies on [audioldm](https://github.com/haoheliu/AudioLDM-training-finetuning). We sincerely appreciate the authors for sharing their code openly.

