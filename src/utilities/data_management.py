import os
import pandas as pd
import yaml

def get_file_list_and_max_number(directory, file_extension):
    """
    List files in the given directory with the given extension and find the maximum number in filenames.
    """
    existing_files = [filename for filename in os.listdir(directory) if filename.endswith(file_extension)]
    existing_numbers = [int(filename.split('_')[-1].split('.')[0]) for filename in existing_files]
    max_number = max(existing_numbers, default=0)
    return existing_files, max_number

def save_to_file(data, directory, base_filename, file_extension, file_number=None):
    """
    Save data to a file, using a specific file number or finding the next available number.
    """
    if file_number is None:
        _, max_number = get_file_list_and_max_number(directory, file_extension)
        file_number = max_number + 1
    filename = f'{directory}{base_filename}_{file_number}{file_extension}'
    if file_extension == '.pkl':
        data.to_pickle(filename)
    elif file_extension == '.yaml':
        with open(filename, 'w') as f:
            yaml.dump(data, f)

def load_from_file(directory, base_filename, file_extension, file_number=None):
    """
    Load data from a file, either the latest or a specific number.
    """
    if file_number is None:
        _, max_number = get_file_list_and_max_number(directory, file_extension)
        file_number = max_number
    filename = f'{directory}{base_filename}_{file_number}{file_extension}'
    if file_extension == '.pkl':
        return pd.read_pickle(filename)
    elif file_extension == '.yaml':
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)