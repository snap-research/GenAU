
# AutoReCap Dataset

# Introduction 
We introduce an efficient pipeline for collecting ambient audio. It starts by analyzing automatic transcriptions of online videos to identify non-speech parts. Our captioning model, AutoCap, then generates captions and filters out segments with music or speech-related keywords. By using time-aligned transcriptions, we reduce the filtering rate and streamline the process by avoiding the need to download or process the audio files.
<br/>

<div align="center">
<img src="../assets/dataset.png" width="1200" />
</div>

<br/>


## Environment Initialization
For initializing your environment, please refer to the [general README](../README.md).

## Autocap Dataset Download
- We currently provide the following datasets:
    * autocap_audioset_vggsounds: containing roughly **445K** audio-text pairs, derived from VGGSounds and a subset of AudioSet. This dataset was not filtered to remove music and speech.
    * AutoReCap-XL: containing around **57M** audio-text pairs, derived from Youtube videos. This dataset contain mainly ambinet audio clips and few speech and music clips. Please refer to the paper for more details on this dataset.

**More datasets will be coming later!**

```shell
python download.py --save_dir <path-to-save-dir> --dataset_name <dataset-subset>

# Example
python download.py --save_dir data/datasets/autocap --dataset_name autocap_audioset_vggsounds --audio_only
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
python organize_dataset.py --save_dir <path-to-dataset> 
                           --dataset_name <key-to-store-dataset> 
                           --split <split-type> 
                           --files_per_subset <number_of_files_per_subset>

# Example
python organize_dataset.py --save_dir data/datasets/autocap --dataset_name autocap --split train
```
- **Important**: Use different dataset_names for different splits.
- If `--files_per_subset` is specified to be more than one, the dataset keys will be named dataset_name_subset_1, dataset_name_subset_2, etc.
- The datasets details can be found at `data/metadata/dataset_root.json`.
- Add the dataset keys under the `data` attribute in your config file for the audio generation and captioning experiments.

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

## Download External Dataset
We provide a script for downloading wavcaps datasets. Run the following scripts to download and organize each of these datasets:

```shell
python download_external_datasets --save_root <path-to-save-root> \
 --dataset_names "dataset_key_1" "dataset_key_2" ...

# Organize each downloaded dataset
python organize_dataset.py --save_dir <path-to-downloaded-dataset> \
 --dataset_name <key-to-store-dataset> 
```
- Available datasets are: **wavcaps_soundbible, wavcaps_bbc, wavcaps_audioset, wavcaps_freesound**
- **Audiocaps and Cloths**: Please refer to the [Audiocaps](https://github.com/cdjkim/audiocaps) and [Clotho](https://zenodo.org/records/3490684) official repositories for instructions on downloading these dataset. We are unable to distribute a copy of the dataset due to copyrights.