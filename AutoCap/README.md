[![arXiv](ARXIV ICON)](ARXIV LINK)

# AutoCap inference, training and evaluation
- [Inference](#inference)
    * [Audio to text script](#audio-to-text)
    * [Gradio demo](#gradio-demo)
    * [Caption a list of audio files](#caption-list-of-audio-files)
    * [Caption your custom dataset](#caption-a-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Cite this work](#cite-this-work)
- [Acknowledgements](#acknowledgements)

# Environment initalization
For initializing your environment, please refer to the [general README](../README.md).

# Inference

## Audio to Text
To quickly generate a caption for an input audio, run
```shell
python scripts/audio_to_text.py --wav_path <path-to-wav-file>

# Example inference
python scripts/audio_to_text.py --wav_path samples/ood_samples/loudwhistle-91003.wav
```
- This will automatically download `TODO` model and run the inference with the default parameters. You may change these parameters or provide your cutome model config file and checkpoint path.
- For more accurate captioning, provide meta data using `--title`, `description`, and `--video_caption` arguments.

## Gradio Demo
A local Gradio demo is also available by running
```shell
python app_audio2text.py
```

## Caption list of audio files
- Prepare all target audio files in a single folder
- Optionally, provide meta data information in `yaml` file using the following structure
```yaml
file_name.wav: 
    title: "video title"
    description: "video description"
    video_caption: "video caption"
```

Then run the following script
```shell
python scripts/inference_folder.py --folder_path <path-to-audio-folder> --meta_data_file <path-to-metadata-yaml-file>

# Example inference
python scripts/inference_folder.py --folder_path samples/ood_samples --meta_data_file samples/ood_samples/meta_data.yaml
```

## Caption your custom dataset

If you want to caption a large dataset, we provide a script that works with multigpus for faster inference.
- Prepare your custom dataset by following the instruction in the dataset prepeartion README (TODO) and run

```shell
python scripts/caption_dataset.py \
            --caption_store_key <key-to-store-generated-captions> \
            --beam_size 2 \
            --start_idx 0 \
            --end_idx 1000000 \
            --dataset_keys "dataset_1" "dataset_2" ...

```
- Provide your dataset keys as registered in the dataset preperation (TODO)
- Captions will be generated and stores in each file json file with the specified caption_ store_key
- `start_idx` and `end_idx` arugments can be used to resume or distribute captioning experiments
- Add your `caption_store_key` under `keys_synonyms:gt_audio_caption` in the target yaml config file for it to be selected when the ground truth caption is not available in your audio captioning or audio generation experiments.


# Training
### Dataset
Please refer to the dataset README (TODO) for instructions on downloading our dataset or preparing your own dataset.

### Stage 1 (pretraining)
- Specify your model parameters in a config yaml file. A sample yaml file is given under `settings/pretraining.yaml`
- Specify your project name and provide your wandb key in the config file. A wandb key can be obtained from [https://wandb.ai/authorize](https://wandb.ai/authorize)
- Optionally, provide your S3 bucket and folder to save intermediate checkpoints. 
- By default, checkpoints will be save under `run_logs/train`
```shell
python train.py -c settings/pretraining.yaml
```

### Stage 2 (finetuning)
- Prepare your finetuning config file in a similar way as the pretraining stage. Typically, you only need to provide `pretrain_path` to your pretraining checkpoint, adjust learning rate, and untoggle the freeze option for the `text_decoder`.
- A sample fintuning config is provided under `settings/finetuning.yaml`

```shell
python train.py -c settings/finetuning.yaml
```


# Evalution
- By default, the models will be log metrics on the validation set to wandb periodically during training as specified in the config file.
- We exclude the `spice`, `spideer` and `meteor` metrics during training as they tend to hang out the training during multigpu training. You man inlcude them by changing the configruation. 
- A file with the predicted captions during evaluation will be saved under `run_logs/train` and metrics can be found in a file named `output.txt` under the logging folder.
- To run the evaluation on the test set, after the training finishes, run:
```shell
python evaluate.py -c <path-to-config> -ckpt <path-to-checkpoint>
```

# Cite this work
If you found this useful, please consider citing our work

```TODO
```

# Acknowledgements
We sincerely thank the authors of the following work for sharing their code publicly:
- [WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research](https://github.com/XinhaoMei/WavCaps)
- [Audio Captioning Transformer](https://github.com/XinhaoMei/ACT/tree/main/coco_caption)
