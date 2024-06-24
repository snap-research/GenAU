import json

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
    json_str = json.dumps(my_dict)
    with open(fname, "w") as json_file:
        json.dump(my_dict, json_file, indent=4)
