import json
import os
import shutil
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
from typing import Optional, Union

# Set up Hugging Face credentials
HF_TOKEN = "add_your_token_here"
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# Vehicle class IDs to filter
VEHICLE_CLASSES = {
    'bus': [654, 779, 874], 
    'car': [436, 468, 627, 717, 751, 817, 867],       
    'truck': [555, 569,867,864],         
    'minivan': [656],
    'bycicle': [444, 671],
    'motorcycle': [665,670],
}

with open('src/dtu_mlops_project/imagenet-simple-labels.json') as f:
    labels = json.load(f)

def class_id_to_label(i):
    return labels[i]
#print(class_id_to_label(924)) # prints "guacamole"

# Flatten class list
target_classes = [cls for classes in VEHICLE_CLASSES.values() for cls in classes]

# And modify the batch filter to be more strict:
def batch_filter(examples):
    """Filter function that only keeps vehicle classes"""
    valid_classes = set()
    for class_list in VEHICLE_CLASSES.values():
        valid_classes.update(class_list)
    
    # Access label from json field
    try:
        labels = [example['label'] for example in examples['json']]
        return [label in valid_classes for label in labels]
    except KeyError as e:
        print(f"Error accessing labels: {e}")
        print("Available keys:", examples.keys())
        return [False] * len(next(iter(examples.values())))

# Define paths
download_path = "data/raw/timm-imagenet-1k-wds"
save_path = "data/processed/filtered_dataset"

# Create directories
os.makedirs(download_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# Load dataset in streaming mode
ds = load_dataset(
    "timm/imagenet-1k-wds",
    split="validation",
    streaming=True,
    cache_dir=download_path
)

# Filter and save in batches
filtered_val = ds.filter(
    batch_filter,
    batched=True,
    batch_size=100
)

# Buffer samples in memory
buffer = []
for sample in filtered_val.take(1000):  # Limit samples
    buffer.append(sample)

# Convert buffer to Dataset
filtered_ds = DatasetDict({
    'validation': Dataset.from_list(buffer)  # Changed from_dict to from_list
})

# Save filtered dataset
filtered_ds.save_to_disk(save_path)

print(f"\nFiltered dataset saved to: {save_path}")
print(f"Validation samples: {len(filtered_ds['validation'])}")

# Add this after creating filtered_ds
def check_dataset_classes(dataset):
    """Print unique classes in dataset"""
    unique_labels = set()
    for sample in dataset['validation']:
        # Access label from json field
        try:
            label = sample['json']['label']
            unique_labels.add(label)
        except KeyError as e:
            print(f"Error accessing label: {e}")
            print("Sample structure:", sample.keys())
            continue
    
    print(f"Unique dataset labels:{sorted(list(unique_labels))}")
    
    # Compare against vehicle classes
    all_valid_classes = set()
    for class_list in VEHICLE_CLASSES.values():
        all_valid_classes.update(class_list)
    print(f"Vehicle class values: {sorted(list(all_valid_classes))}")
    
    # Show unexpected classes
    unexpected = unique_labels - all_valid_classes
    if unexpected:
        print(f"WARNING: Found unexpected classes: {sorted(list(unexpected))}")

    return unique_labels

check_dataset_classes(filtered_ds)



def show_subset_samples(
    output_dir: Union[str, DatasetDict], 
    save_path: str,
    n_samples: int = 9,
    seed: int = 42,
    split: Optional[str] = None
) -> str:
    """Display sample images from dataset and save to file."""
    splits = ['train', 'validation'] if split is None else [split]
    # Set seed for reproducibility
    random.seed(seed)
    fig = plt.figure(figsize=(12, 12))
    for split_name in splits:
        if isinstance(output_dir, DatasetDict):
            dataset = output_dir[split_name]
            samples = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
            for i, idx in enumerate(samples, 1):
                ax = plt.subplot(3, 3, i)
                try:
                    img = dataset[idx]['jpg']
                    label = dataset[idx]['json']['label']
                    filename = dataset[idx]['json']['filename']
                    classname = class_id_to_label(label)
                    plt.imshow(img)
                    plt.title(f"Class {label}: {classname}")
                    plt.suptitle(f'Samples from {split_name}', fontsize=16)
                    plt.axis('off')
                except KeyError as e:
                    print(f"Missing key in dataset: {e}")
                    print(f"Available keys: {dataset[idx].keys()}")
                    raise
        else:
            # Handle filesystem path input
            split_dir = Path(output_dir) / split_name
            image_files = list(split_dir.glob('**/*.JPEG'))
            samples = random.sample(image_files, min(n_samples, len(image_files)))
            for i, img_path in enumerate(samples, 1):
                ax = plt.subplot(3, 3, i)
                img = Image.open(img_path)
                plt.imshow(img)
                plt.title(f"{split_name}: {img_path.parent.name}")
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# Verify save worked by trying to load
try:
    loaded_ds = load_from_disk(save_path)
    print(f"\nSuccessfully saved and verified dataset at: {save_path}")
    #print(f"Loaded dataset size: {len(loaded_ds['train'])} train, "
     #     f"{len(loaded_ds['validation'])} validation, ")
          #f"{len(loaded_ds['test'])} test samples")
except Exception as e:
    print(f"Error verifying saved dataset: {e}")

# Usage example:
#loaded_ds = load_from_disk(save_path)
saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed40.png", split='validation', seed=40)
saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed41.png", split='validation', seed=41)
saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed42.png", split='validation', seed=42)
saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed43.png", split='validation', seed=43)
saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed44.png", split='validation', seed=44)
saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed45.png", split='validation', seed=45)
#- [Classes](https://huggingface.co/datasets/ILSVRC/imagenet-1k/blob/main/classes.py)
#   - Trucks:
#   - Buses: n03769881, n04146614, n04487081
#   - Cars: n04285008
#   - Motorcycles:
#   - Bicycles:
#   - Tractor: n04465501



