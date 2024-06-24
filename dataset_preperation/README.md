
# AutoCap Dataset Preparation

## Environment Initialization
For initializing your environment, please refer to the [general README](../README.md).

## Dataset Download
- We currently provide the following datasets:
    * autocap_audioset_vggsounds: containing **444,837** audio-text pairs.

**More datasets will be coming soon!**

```shell
python download.py --save_dir <path-to-save-dir> --dataset_name <dataset-subset>

# Example
python download.py --save_dir data/autocap --dataset_name autocap_audioset_vggsounds --audio_only
```
By default, the script will download videos along with their metadata.

We provide the following helpful arguments:
- `--sampling_rate`: Specifies the sampling rate at which the audio files are to be stored.
- `--audio_only`: Download only the audio files and discard the videos. This is helpful to save storage space.
- `--files_per_folder`: Downloaded files will be organized into many folders. This argument specifies how many files to store per folder.
- `--start_idx`, `--end_idx`: To download only a subset of the dataset.
- `--proxy`: For large downloads, YouTube might block your address. You may SSH to another machine at a specific port and provide it using this argument.

## Dataset Organization
Once the dataset finishes downloading, run the following script:
```shell
python organize_dataset.py --save_dir <path-to-dataset> --dataset_name <key-to-store-dataset> --split <split-type> --files_per_subset <number_of_files_per_subset>

# Example
python organize_dataset.py --save_dir data/autocap --dataset_name autocap --split train
```
- If `--files_per_subset` is specified to be more than one, the dataset keys will be named as dataset_name_subset_1, dataset_name_subset_2, etc.
- The datasets details can be found at `data/metadata/dataset_root.json`.
- Add the dataset keys under  the`data` column in your config file for audio generation and captioning experiments.

## Prepare Your Custom Dataset
You need to arrange your audio files in one folder using the following structure:
```
- Folder
    - 000000
        - Id_1.wav
        - Id_1.json
        - Id_2.wav
        - Id_2.json
    - 000001
        - Id_3.wav
        - Id_3.json
        .
        .
```
- In the JSON files, add the metadata such as title, description, video_caption, and gt_audio_caption.
- Organizing your dataset following the instructions in [Dataset Organization](#dataset-organization).
