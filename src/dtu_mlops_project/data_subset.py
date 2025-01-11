import json
#import requests
import os
import shutil
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
import numpy as np

# Set up Hugging Face credentials
HF_TOKEN = "add_your_token_here"
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

#url = "https://datasets-server.huggingface.co/rows"
#params = {
#    "dataset": "timm/mini-imagenet", # timm/imagenet-1k-wds
#    "config": "default",
#    "split": "train", # remember to change this to 'validation' 
#    "offset": "0",
#    "length": "100"
#}
#headers = {
#    "Authorization": f"Bearer {HF_TOKEN}"
#}
#response = requests.get(url, params=params, headers=headers)
##print(response.json())
#
## Get the response data
#data = response.json()
#
## Find and print the row where __key__ matches
#target_key = "n02097047_2079"
#for row in data.get('rows', []):
#    if row.get('row', {}).get('__key__') == target_key:
#        print("Found row with key", target_key)
#        print(json.dumps(row['row'], indent=2))
#
#
## JSON Structure
#json_string = '''
#{
#    "features": [
#        {"feature_idx": 0, "name": "__key__", "type": {"dtype": "string", "_type": "Value"}},
#        {"feature_idx": 1, "name": "__url__", "type": {"dtype": "string", "_type": "Value"}},
#        {"feature_idx": 2, "name": "cls", "type": {"dtype": "int64", "_type": "Value"}},
#        {"feature_idx": 3, "name": "jpg", "type": {"_type": "Image"}},
#        {"feature_idx": 4, "name": "json", "type": {
#            "filename": {"dtype": "string", "_type": "Value"},
#            "height": {"dtype": "int64", "_type": "Value"},
#            "label": {"dtype": "int64", "_type": "Value"},
#            "width": {"dtype": "int64", "_type": "Value"}
#        }}
#    ]
#}
#'''
#
## Parse the JSON string into a Python object
#parsed_data = json.loads(json_string)
#
## Access specific parts of the data
#features = parsed_data["features"]

# Print the number of features
#print(f"Number of features: {len(features)}")

# Print details of each feature
#for i, feature in enumerate(features):
#    print(f"\nFeature {i + 1}:")
#    for key, value in feature.items():
#        print(f"{key}: {value}")

# Example: Get the type of the 'jpg' feature
#jpg_type = features[3]["type"]
#print("\nType of 'jpg' feature:", jpg_type)








# Define the custom download path (relative path)
download_path = "data/raw/timm-imagenet-1k-wds"
#download_path = "data/raw/timm/mini-imagenet"

# Create the folder if it does not exist
os.makedirs(download_path, exist_ok=True)

# Load dataset 
ds = load_dataset("timm/imagenet-1k-wds", split="validation",cache_dir=download_path)
#ds = load_dataset("timm/mini-imagenet",  cache_dir=download_path)

# Access metadata
print("Dataset metadata:")
print(ds)

def batch_filter(subset):
    return {'keep': np.array(subset['cls']) == 1}

# Filter datasets using batch operations
#filtered_train = ds['train'].filter(
#    batch_filter,
#    batched=True,
#    batch_size=1000,
#    num_proc=4,
#    cache_file_name=".cache/filtered_train"
#)

filtered_val = ds['validation'].filter(
    batch_filter, 
    batched=True,
    batch_size=1000,
    num_proc=4,
    cache_file_name=".cache/filtered_val"
)

# Create new DatasetDict with filtered datasets
filtered_ds = DatasetDict({
    #'train': filtered_train,
    'validation': filtered_val#,'test': filtered_test
})

# Print statistics
print("\nFiltered Dataset Statistics:")
#print(f"Train samples: {len(filtered_ds['train'])}")
print(f"Validation samples: {len(filtered_ds['validation'])}")
#print(f"Test samples: {len(filtered_ds['test'])}")

# Print full dataset info
print("\nFiltered Dataset Info:")
print(filtered_ds)

# Define save path
save_path = "data/processed/filtered_dataset"

# Create output directory
os.makedirs(save_path, exist_ok=True)

# Save filtered dataset to disk
filtered_ds.save_to_disk(save_path)

# Verify save worked by trying to load
try:
    loaded_ds = load_from_disk(save_path)
    print(f"\nSuccessfully saved and verified dataset at: {save_path}")
    #print(f"Loaded dataset size: {len(loaded_ds['train'])} train, "
     #     f"{len(loaded_ds['validation'])} validation, ")
          #f"{len(loaded_ds['test'])} test samples")
except Exception as e:
    print(f"Error verifying saved dataset: {e}")

# Remove cache directories
try:    
    cache_dirs = [
        "_00000_of_00004.cache",
        "_00001_of_00004.cache", 
        "_00002_of_00004.cache",
        "_00003_of_00004.cache"
    ]

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Removed cache directory: {cache_dir}")
except Exception as e:
    print(f"Error removing cache directories: {e}")

#- [Classes](https://huggingface.co/datasets/ILSVRC/imagenet-1k/blob/main/classes.py)
#   - Trucks:
#   - Buses: n03769881, n04146614, n04487081
#   - Cars: n04285008
#   - Motorcycles:
#   - Bicycles:
#   - Tractor: n04465501



