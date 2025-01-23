import json
import os
from pathlib import Path
from typing import Any
import typer
from loguru import logger
from datasets import Dataset, DatasetDict, load_dataset
from dataclasses import dataclass

app = typer.Typer()

# Set up Huggingface credentials using python locally or use docker -e to pass the HUGGING_FACE_HUB_TOKEN
# HF_TOKEN = "add_your_token_here"
# os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# Paths and configuration
@dataclass
class Config:
    download_path: Path = Path("data/raw/timm-imagenet-1k-wds")
    processed_path: Path = Path("data/processed/timm-imagenet-1k-wds-subset/")
    config_path: Path = Path("configs/vehicle_classes.json")
    labels_path: Path = Path("src/dtu_mlops_project/imagenet-simple-labels.json")


config = Config()


def load_vehicle_classes():
    with open(config.config_path) as f:
        vehicle_classes = json.load(f)
    return vehicle_classes


# Load vehicle classes from JSON
VEHICLE_CLASSES = load_vehicle_classes()

# Flatten class list
target_classes = [cls for classes in VEHICLE_CLASSES.values() for cls in classes]

# Load class labels
# Source: https://github.com/anishathalye/imagenet-simple-labels
with open(config.labels_path) as f:
    labels = json.load(f)


def class_id_to_label(i):
    return labels[i]


# batch filter function
def batch_filter(examples: dict[str, Any]) -> list[bool]:
    """Filter function that only keeps classes defined in config_path JSON-file."""
    valid_classes: set[str] = set()
    for class_list in VEHICLE_CLASSES.values():
        valid_classes.update(class_list)

    # Access label from json field
    try:
        labels = [example["label"] for example in examples["json"]]
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
                label = sample["json"]["label"]
                # label_name = class_id_to_label(label)

                # Create class directory
                class_path = os.path.join(base_path, str(label))
                os.makedirs(class_path, exist_ok=True)

                # Save image directly since it's already a PIL Image
                image_path = os.path.join(class_path, f"{idx}.jpg")
                sample["jpg"].save(image_path)

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
            label = sample["json"]["label"]
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
    split: str, download_path: str, processed_path: str, buffer_size: int = 10000, batch_size: int = 100
) -> tuple[DatasetDict, dict]:
    """Process a single dataset split.

    Args:
        split: Dataset split name ('train' or 'validation')
        download_path: Path to store raw downloaded data
        processed_path: Path to store processed data
        buffer_size: Number of images to buffer in memory, at a minimum set to 1500 * number of classes
        batch_size: Batch size for filtering

    Returns:
        tuple of (filtered dataset, label counts)
    """
    # Add validation
    if split not in ["train", "validation"]:
        raise ValueError("Split must be either 'train' or 'validation'")
    if buffer_size < 1:
        raise ValueError("Buffer size must be positive")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    # Set paths
    save_path = f"data/raw/filtered_dataset/{split}"
    processed_path_split = os.path.join(processed_path, split)

    # Create directories
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(processed_path_split, exist_ok=True)

    # Load dataset in streaming mode
    ds = load_dataset("timm/imagenet-1k-wds", split=split, streaming=True, cache_dir=download_path)

    # Filter and save in batches
    filtered_val = ds.filter(batch_filter, batched=True, batch_size=batch_size)

    # Buffer samples
    buffer = []
    for sample in filtered_val.take(buffer_size):
        buffer.append(sample)

    # Convert to Dataset
    filtered_ds = DatasetDict({split: Dataset.from_list(buffer)})

    # Count samples per label
    label_counts = {}
    for sample in filtered_ds[split]:
        label = sample["json"]["label"]
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
    splits: list[str] = typer.Option(["train", "validation"], help="Dataset splits to process"),
    download_path: Path = typer.Option("data/raw/timm-imagenet-1k-wds", help="Path to store downloaded data"),
    processed_path: Path = typer.Option(
        "data/processed/timm-imagenet-1k-wds-subset", help="Path to store processed data"
    ),
    buffer_size: int = typer.Option(1000, help="Number of samples to buffer"),
    batch_size: int = typer.Option(100, help="Batch size for filtering"),
) -> None:
    """Process ImageNet dataset splits into class subset."""
    for split in splits:
        filtered_ds, label_counts = process_dataset_split(
            split=split,
            download_path=str(download_path),
            processed_path=str(processed_path),
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        logger.info(f"Processed {split} split:")
        logger.info(f"Samples saved to: {processed_path}/{split}")
        logger.info(f"Total samples: {len(filtered_ds[split])}")
        logger.info("Samples per class:")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  {label}: {count}")

    check_dataset_classes(filtered_ds, split)


if __name__ == "__main__":
    app()
