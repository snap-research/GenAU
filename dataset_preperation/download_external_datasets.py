from pathlib import Path
import os
import argparse
import subprocess
import json
import ast
from huggingface_hub import snapshot_download
from tqdm import tqdm

def update_wavcaps_json_files(base_path, central_json_path, blacklist_ids=[], autocap_captions={}):
    # Load the central JSON with directory data
    with open(central_json_path, 'r') as file:
        dirs = json.load(file)['data']

    # Loop through each entry in the central JSON
    for entry in tqdm(dirs):
        id = entry['id']
        if id.endswith('.wav'):
            id = id[:-4]
        
        # skip blacklisted ids
        if id in blacklist_ids:
            continue
        
        json_file_path = os.path.join(base_path, f"{id}.json")
        falc_path = os.path.join(base_path, f"{id}.flac")
        
        # Check if the corresponding id.json file exists
        if os.path.exists(falc_path):
            data = {}

            data['wav'] = falc_path
            # Convert category from stringified list to list
            if 'category' in entry:
                try:
                    category_list = ast.literal_eval(entry['category'])
                except ValueError:
                    category_list = entry['category']  # Fallback to original if conversion fails
                data['category'] = category_list
            
            if 'tags' in entry:
                try:
                    category_list = ast.literal_eval(entry['tags'])
                except ValueError:
                    category_list = entry['tags']  # Fallback to original if conversion fails
                data['tags'] = category_list
            
            data['wavcaps_caption'] = entry['caption']
            keys = ['download_link', 'id', 'download_link', 'title', 'description']
            for k in keys:
                if k in entry:
                    data[k] = entry[k]
            
            
            # add autocap caption if exists
            if f'{id}.json' in autocap_captions:
                data['autocap_caption'] = autocap_captions[f'{id}.json']
                
            # Save the updated data back to the file
            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=4)
            
def download_and_organize_wavcaps(dataset_root, subset_key, autocap_captions_file_path='data/json_files/wavcaps_autocap_captions.json'):
    # read autocap captions
    with open(autocap_captions_file_path, 'r') as f:
        autocap_captions = json.load(f)
        
    dataset_dir = os.path.join(dataset_root, 'wavcaps')
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    
    # unzip the files 
    # soundbible
    if subset_key == 'wavcaps_soundbible':
        print("[INFO] Processing Wavcaps Soundbible subset")
        snapshot_download(repo_id="cvssp/WavCaps", local_dir=dataset_dir, repo_type="dataset", allow_patterns=['Zip_files/SoundBible/*', 'json_files/*'])
        os.system(f"unzip {os.path.join(dataset_dir, 'Zip_files/SoundBible/SoundBible.zip')} -d {os.path.join(dataset_dir, 'wavcaps_soundbible')}")
        base_path = os.path.join(dataset_dir, "wavcaps_soundbible", "mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/SoundBible_flac")
        central_json_path = os.path.join(dataset_dir, "json_files/SoundBibl0e/sb_final.json")
        update_wavcaps_json_files(base_path, central_json_path, autocap_captions=autocap_captions['Soundbible'])
    
    # bbc
    elif subset_key == 'wavcaps_bbc':
        print("[INFO] Processing Wavcaps BBC Sound Effects subset")
        snapshot_download(repo_id="cvssp/WavCaps", local_dir=dataset_dir, repo_type="dataset", allow_patterns=['Zip_files/BBC_Sound_Effects/*', 'json_files/*'])
        os.system(f"zip -F {os.path.join(dataset_dir, 'Zip_files/BBC_Sound_Effects/BBC_Sound_Effects.zip')} --out {os.path.join(dataset_dir, 'Zip_files/BBC_Sound_Effects/bbc.zip')}")
        os.system(f"unzip {os.path.join(dataset_dir, 'Zip_files/BBC_Sound_Effects/bbc.zip')} -d {os.path.join(dataset_dir, 'wavcaps_bbc')}")
        base_path =  os.path.join(dataset_dir, "wavcaps_bbc", "mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/BBC_Sound_Effects_flac")
        central_json_path = os.path.join(dataset_dir, "json_files/BBC_Sound_Effects/bbc_final.json")
        update_wavcaps_json_files(base_path, central_json_path, autocap_captions=autocap_captions['BBC'])
    
    # audioset
    elif subset_key == 'wavcaps_audioset':
        print("[INFO] Processing Wavcaps Audioset subset")
        snapshot_download(repo_id="cvssp/WavCaps", local_dir=dataset_dir, repo_type="dataset", allow_patterns=['Zip_files/AudioSet_SL/*', 'json_files/*'])
        os.system(f"zip -F {os.path.join(dataset_dir, 'Zip_files/AudioSet_SL/AudioSet_SL.zip')} --out {os.path.join(dataset_dir, 'Zip_files/AudioSet_SL/AS.zip')}")
        os.system(f"unzip {os.path.join(dataset_dir, 'Zip_files/AudioSet_SL/AS.zip')} -d {os.path.join(dataset_dir, 'wavcaps_audioset')}")
        base_path = os.path.join(dataset_dir, "wavcaps_audioset", "mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac")
        central_json_path = os.path.join(dataset_dir, "json_files/AudioSet_SL/as_final.json")
        
        # blacklist
        with open(os.path.join(dataset_dir, 'json_files/blacklist/blacklist_exclude_all_ac.json'), 'r') as f:
            blacklist = json.load(f)
        blacklist_ids = set([id.split('.')[0] for id in blacklist['AudioSet']])
        update_wavcaps_json_files(base_path, central_json_path, blacklist_ids=blacklist_ids, autocap_captions=autocap_captions['AudioSet'])
    
    # freesound
    elif subset_key == 'wavcaps_freesound':
        print("[INFO] Processing Wavcaps Freesound subset")
        snapshot_download(repo_id="cvssp/WavCaps", local_dir=dataset_dir, repo_type="dataset", allow_patterns=['Zip_files/FreeSound/*', 'json_files/*'])
        os.system(f"zip -F {os.path.join(dataset_dir, 'Zip_files/FreeSound/FreeSound.zip')} --out {os.path.join(dataset_dir, 'Zip_files/FreeSound/freesound.zip')}")
        os.system(f"unzip {os.path.join(dataset_dir, 'Zip_files/FreeSound/freesound.zip')} -d {os.path.join(dataset_dir, 'wavcaps_freesound')}")
        base_path = os.path.join(dataset_dir, "wavcaps_freesound", "mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/FreeSound_flac")
        central_json_path = os.path.join(dataset_dir, "json_files/FreeSound/fsd_final.json")
        
        with open(os.path.join(dataset_dir, 'json_files/blacklist/blacklist_exclude_all_ac.json'), 'r') as f:
            blacklist = json.load(f)
        blacklist_ids = set([id.split('.')[0] for id in blacklist['FreeSound']])
        
        update_wavcaps_json_files(base_path, central_json_path, blacklist_ids=blacklist_ids, autocap_captions=autocap_captions['FreeSound'])
    


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_names", 
                        type=str,
                        nargs='*',
                        required=True,
                        help=f"Provided the dataset names. Available datasets are [audiocaps, clotho, wavcaps_soundbible, wavcaps_bbc, wavcaps_audioset, wavcaps_freesound]")

parser.add_argument("--save_root", 
                    type=str,
                    required=False,
                    default='data/datasets/',
                    help="Where to save the downloaded files")
    
args = parser.parse_args()

for dataset_key in args.dataset_names:
    if dataset_key == 'clotho':
        download_and_organize_clotho(args.save_root)
    
    elif dataset_key == 'audiocaps':
        download_and_organize_audiocaps(args.save_root)
    
    elif dataset_key.startswith('wavcaps'):
        download_and_organize_wavcaps(args.save_root, dataset_key)
    
    else:
        raise KeyError(f"Invalid dataset key {dataset_key}")