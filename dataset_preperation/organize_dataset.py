import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool, get_context
import logging
from io import StringIO
import json
import argparse
from pathlib import Path


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Attempt to read the JSON data

    except json.JSONDecodeError as e:
        with open(file_path, 'r') as file:
            # Read the file content till the point where JSON is valid
            file_content = file.read()
            valid_json = file_content[:file_content.rfind('}')+1]

            try:
                data = json.loads(valid_json)  # Reload the valid JSON part
            except json.JSONDecodeError:
                print("Failed to recover JSON.")
                return None

        # Save the cleaned JSON data to a new file
        if data is not None:
            with open(file_path, 'w') as new_file:
                json.dump(data, new_file, indent=4)
    return data

    
def load_file(fname):
    with open(fname, "r") as f:
        return f.read().split('\n')[:-1]

def write_json(my_dict, fname):
    with open(fname, "w") as json_file:
        json.dump(my_dict, json_file, indent=4)
        
def find_json_files(directory):
    json_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return json_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", 
                        type=str,
                        required=True,
                        help="where to save the downloaded files")
    
    parser.add_argument("--dataset_meta_file", 
                        required=False,
                        type=str,
                        default='data/metadata/dataset_root.json',
                        help="path to the dataset root json file where the datafiles paths will be stores")
    
    parser.add_argument("--datafiles_dir", 
                        required=False,
                        type=str,
                        default='data/metadata/datafiles/autocap',
                        help="directories where the datafiles will be stored")
    
    parser.add_argument("--dataset_name", 
                        type=str,
                        default='autocap',
                        help="Name of the compiled dataset")
    
    parser.add_argument("--files_per_subset", 
                        type=int,
                        default=-1,
                        help="How many files to include in each subset. -1 put all files in a single subset")
    parser.add_argument("--split", 
                        type=str,
                        default='train',
                        help="split of the dataset")
    
    parser.add_argument("--overwrite", 
                        default=False,
                        action="store_true"
                        help="Overwrite dataset metadata")
    
    args = parser.parse_args()
    
    # initialize all paths
    Path(args.datafiles_dir).mkdir(parents=True, exist_ok=True)
    Path(args.dataset_meta_file).parent.mkdir(parents=True, exist_ok=True)

    # find all .json files
    all_json_files = find_json_files(args.save_dir)
    
    current_subset = 1
    current_dataset_name = f"{args.dataset_name}_subset_{current_subset}" if args.files_per_subset > 0 else args.dataset_name
    current_datafile_path = os.path.join(args.datafiles_dir, f"{current_dataset_name}_{args.split}.txt")
    current_datafile = open(current_datafile_path, 'w')
    
    all_datafiles_path = [(args.split, current_dataset_name, current_datafile_path)]
    for idx, file_path in enumerate(all_json_files):
        current_datafile.write(f"{os.path.relpath(file_path, args.save_dir)}\n")
        
        if args.files_per_subset > 0 and (idx + 1) % args.files_per_subset == 0 and (idx+1) < len(all_json_files):
            current_subset += 1
            current_dataset_name = f"{args.dataset_name}_subset_{current_subset}" 
            
            # close current file and open a new one
            current_datafile.close()
            current_datafile_path = os.path.join(args.datafiles_dir, f"{current_dataset_name}_{args.split}.txt")
            all_datafiles_path.append((args.split, current_dataset_name, current_datafile_path))
            current_datafile = open(current_datafile_path, 'w')
            
    current_datafile.close()
    
    # write on the dataset root files
    if os.path.exists(args.dataset_meta_file):
        dataset_root = load_json(args.dataset_meta_file)
    else:
        dataset_root = {"metadata":{"path":{}}}
    
    # add all datasets
    for split, dataset_name, datafile_path in all_datafiles_path:
        if not args.overwrite and dataset_name in dataset_root:
            raise ValueError("ERROR: {dataset_name} already exists in {args.dataset_meta_file}, please use a different dataset_name or pass --overwrite")
        
        dataset_root[dataset_name] = os.path.abspath(args.save_dir)
        dataset_root['metadata']['path'][dataset_name] = dataset_root['metadata']['path'].get(dataset_name, {})
        dataset_root['metadata']['path'][dataset_name][split] = os.path.abspath(datafile_path)
        for split_check in ['train', 'test', 'val']:
            dataset_root['metadata']['path'][dataset_name][split_check] = dataset_root['metadata']['path'][dataset_name].get(split_check, "")
    
    write_json(dataset_root, args.dataset_meta_file)
    print("[INFO] Congrats! done organizing dataset")
    print("[INFO] Please use the following file path as the `metadata_root` in your experiments configurations:", os.path.abspath(args.dataset_meta_file))
    print("[INFO] You may use any of the following datasets to use in your experiments are:", [entry[1] for entry in all_datafiles_path])
        