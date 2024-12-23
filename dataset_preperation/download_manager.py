import os
import wget

save_dir = 'data/json_files'
dataset_urls = {"AutoReCapXL":'https://huggingface.co/datasets/mali6/autocap/resolve/main/processed_snap-hdvila100m-w-clap-videos_segments_filtered_above_01.json',
                "AutoReCapXL-MQ": 'https://huggingface.co/datasets/mali6/autocap/resolve/main/processed_snap-hdvila100m-w-clap-videos_segments_filtered_above_04.json',
                "AutoReCapXL-MQ-L": 'https://huggingface.co/datasets/mali6/autocap/resolve/main/processed_snap-hdvila100m-w-clap-videos_segments_filtered_above_04_longer_than_5s.json',
                "AutoReCapXL-HQ": 'https://huggingface.co/datasets/mali6/autocap/resolve/main/processed_snap-hdvila100m-w-clap-videos_segments_filtered_above_05.json'}
                

def get_dataset_json_file(dataset_name, dataset_json_file_path=None, download=True):
    if dataset_json_file_path is None:
        dataset_json_file_path = os.path.join(save_dir, f"{dataset_name}.json")
    if os.path.exists(dataset_json_file_path):
        return dataset_json_file_path
    elif not download:
        raise f"[ERROR] Dataset json file does not exist at {dataset_json_file_path}, please use download flag to attempt to downloaded it from the web or manually download it from https://huggingface.co/datasets/mali6/autocap/"
    else:
        os.makedirs(save_dir, exist_ok=True)
        if dataset_name not in dataset_urls:
            raise f"[ERROR] Dataset {dataset_name} is not recognized and its json file does not exist at {dataset_json_file_path}"
        wget.download(dataset_urls[dataset_name], dataset_json_file_path)
        print(f"[INFO] JSON file for dataset {dataset_name} is downloaded at {dataset_json_file_path}")
        return dataset_json_file_path
        