import json
import os
import random
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import typer
from loguru import logger
from PIL import Image

from datasets import Dataset, DatasetDict, load_dataset

app = typer.Typer()


# Define paths
download_path = "data/raw/timm-imagenet-1k-wds"
processed_path = "data/processed/timm-imagenet-1k-wds-subset/"

# Vehicle class IDs to filter
VEHICLE_CLASSES = {
    'bus': [654 ], # 874, 779
    'car': [436], #468, 627, 717, 751, 817, 867
    'truck': [555], # 867, 569, 864   
    #'minivan': [656],
    #'bicycle': [444, 671],
    #'motorcycle': [665,670],
}

# Flatten class list
target_classes = [cls for classes in VEHICLE_CLASSES.values() for cls in classes]

# Load class labels
# Source: https://github.com/anishathalye/imagenet-simple-labels
with open('src/dtu_mlops_project/imagenet-simple-labels.json') as f:
    labels = json.load(f)
def class_id_to_label(i):
    return labels[i]
#logger.info(class_id_to_label(654))

# batch filter function
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
        logger.error(f"Error accessing labels: {e}")
        logger.error("Available keys:", examples.keys())
        return [False] * len(next(iter(examples.values())))

# Save images to files
def save_images_to_files(dataset_dict, base_path):
    """Save dataset images to directory structure."""
    for split, dataset in dataset_dict.items():
        
        # Save each image
        for idx, sample in enumerate(dataset):
            try:
                # Get class label
                label = sample['json']['label']
                #label_name = class_id_to_label(label)
                
                # Create class directory
                class_path = os.path.join(base_path, str(label))
                os.makedirs(class_path, exist_ok=True)
                
                # Save image directly since it's already a PIL Image
                image_path = os.path.join(class_path, f"{idx}.jpg")
                sample['jpg'].save(image_path)
                
            except Exception as e:
                logger.error(f"Error saving image {idx}: {e}")
                continue

# Add this after creating filtered_ds
def check_dataset_classes(dataset, split):
    """logger.info unique classes in dataset"""
    unique_labels = set()
    for sample in dataset[split]:
        # Access label from json field
        try:
            label = sample['json']['label']
            unique_labels.add(label)
        except KeyError as e:
            logger.error(f"Error accessing label: {e}")
            logger.error("Sample structure:", sample.keys())
            continue
    
    logger.info(f"Unique dataset labels:{sorted(list(unique_labels))}")
    
    # Compare against vehicle classes
    all_valid_classes = set()
    for class_list in VEHICLE_CLASSES.values():
        all_valid_classes.update(class_list)
    logger.info(f"Vehicle class values: {sorted(list(all_valid_classes))}")
    
    # Show unexpected classes
    unexpected = unique_labels - all_valid_classes
    if unexpected:
        logger.info(f"WARNING: Found unexpected classes: {sorted(list(unexpected))}")

    return unique_labels

@app.command()
# Process a single dataset split
def process_dataset_split(
    split: str,
    download_path: str,
    processed_path: str,
    buffer_size: int = 1000,
    batch_size: int = 100
) -> tuple[DatasetDict, dict]:
    """Process a single dataset split.
    
    Args:
        split: Dataset split name ('train' or 'validation')
        download_path: Path to store raw downloaded data
        processed_path: Path to store processed data
        buffer_size: Number of samples to buffer in memory
        batch_size: Batch size for filtering
        
    Returns:
        tuple of (filtered dataset, label counts)
    """
    # Set paths
    save_path = f"data/raw/filtered_dataset/{split}"
    processed_path_split = os.path.join(processed_path, split)

    # Create directories
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(processed_path_split, exist_ok=True)
    
    # Load dataset in streaming mode
    ds = load_dataset(
        "timm/imagenet-1k-wds",
        split=split,
        streaming=True,
        cache_dir=download_path
    )

    # Filter and save in batches
    filtered_val = ds.filter(
        batch_filter,
        batched=True,
        batch_size=batch_size
    )

    # Buffer samples
    buffer = []
    for sample in filtered_val.take(buffer_size):
        buffer.append(sample)

    # Convert to Dataset
    filtered_ds = DatasetDict({
        split: Dataset.from_list(buffer)  
    })

    # Count samples per label
    label_counts = {}
    for sample in filtered_ds[split]:
        label = sample['json']['label']
        label_name = class_id_to_label(label)
        label_counts[label_name] = label_counts.get(label_name, 0) + 1

    # Save filtered dataset
    filtered_ds.save_to_disk(save_path)
    
    # Save images
    save_images_to_files(filtered_ds, processed_path_split)

    return filtered_ds, label_counts

@app.command()
# Process multiple dataset splits
def process_splits(
    splits: list[str] = typer.Option(
        ["train", "validation"], 
        help="Dataset splits to process"
    ),
    download_path: Path = typer.Option(
        "data/raw/timm-imagenet-1k-wds",
        help="Path to store downloaded data"
    ),
    processed_path: Path = typer.Option(
        "data/processed/timm-imagenet-1k-wds-subset",
        help="Path to store processed data"
    ),
    buffer_size: int = typer.Option(
        1000,
        help="Number of samples to buffer"
    ),
    batch_size: int = typer.Option(
        100,
        help="Batch size for filtering"
    )
) -> None:
    """Process ImageNet dataset splits into vehicle class subset."""
    for split in splits:
        filtered_ds, label_counts = process_dataset_split(
            split=split,
            download_path=str(download_path),
            processed_path=str(processed_path),
            buffer_size=buffer_size,
            batch_size=batch_size
        )
        
        logger.info(f"Processed {split} split:")
        logger.info(f"Samples saved to: {processed_path}/{split}")
        logger.info(f"Total samples: {len(filtered_ds[split])}")
        logger.info("Samples per class:")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  {label}: {count}")


    check_dataset_classes(filtered_ds, split)



    #save_images_to_files(filtered_ds, processed_path_split)
# display samples
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
                    plt.title(f"Class {label}: {classname.capitalize()}")
                    #plt.title(f"Class {label}: {classname.capitalize()}\n{filename}") # Option to show filename
                    plt.suptitle(f'Samples from {split_name}', fontsize=16)
                    plt.axis('off')
                except KeyError as e:
                    logger.error(f"Missing key in dataset: {e}")
                    logger.error(f"Available keys: {dataset[idx].keys()}")
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


## Verify save worked by trying to load
#try:
#    loaded_ds = load_from_disk(save_path)
#    logger.info(f"\nSuccessfully saved and verified dataset at: {save_path}")
#except Exception as e:
#    logger.error(f"Error verifying saved dataset: {e}")
#
## Plot samples:
#saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed40_"+split+".png", split=split, seed=40)
#saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed41_"+split+".png", split=split, seed=41)
#saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed42_"+split+".png", split=split, seed=42)
#saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed43_"+split+".png", split=split, seed=43)
#saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed44_"+split+".png", split=split, seed=44)
#saved_fig = show_subset_samples(output_dir=loaded_ds, save_path="reports/figures/data_samples_seed45_"+split+".png", split=split, seed=45)




if __name__ == "__main__":
    app()