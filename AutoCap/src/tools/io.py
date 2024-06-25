#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

from pathlib import Path
import os
import csv
import pickle
import json
import yaml

def write_csv_file(csv_obj, file_name):
    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, csv_obj[0].keys())
        writer.writeheader()
        writer.writerows(csv_obj)
    print(f'Write to {file_name} successfully.')


def load_csv_file(file_name):
    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj

def load_yaml_file(file_name):
    with open(file_name, "r") as f:
        yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_obj

def load_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj


def write_pickle_file(obj, file_name):
    Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Write to {file_name} successfully.')

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
        