import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from datasets import Dataset, DatasetDict
from src.dtu_mlops_project.data import (
    Config, 
    save_images_to_files,
    process_dataset_split
)

@pytest.fixture
def config():
    return Config(
        download_path=Path("test_data/raw"),
        processed_path=Path("test_data/processed"),
        config_path=Path("test_data/configs/vehicle_classes.json"),
        labels_path=Path("test_data/labels.json")
    )

@pytest.fixture
def mock_vehicle_classes():
    return {
        "cars": ["car", "sports_car"],
        "trucks": ["pickup", "truck"]
    }

@pytest.fixture
def mock_dataset():
    # Create synthetic image data
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    
    data = {
        "json": [
            {"label": "car"},
            {"label": "truck"},
            {"label": "minibus"}  # Invalid class
        ],
        "jpg": [image, image, image]
    }
    return DatasetDict({
        "train": Dataset.from_dict(data)
    })

def test_config_initialization(config):
    assert isinstance(config.download_path, Path)
    assert isinstance(config.processed_path, Path)
    assert isinstance(config.config_path, Path)
    assert isinstance(config.labels_path, Path)

def test_process_dataset_split(tmp_path):
    split = "train"
    download_path = str(tmp_path / "raw")
    processed_path = str(tmp_path / "processed")
    
    with pytest.raises(ValueError):
        # Should fail with invalid split name
        process_dataset_split("invalid_split", download_path, processed_path)
        
    with pytest.raises(ValueError):
        # Should fail with invalid buffer size
        process_dataset_split(split, download_path, processed_path, buffer_size=-1)

def test_save_images_to_files(mock_dataset, tmp_path):
    save_images_to_files(mock_dataset, tmp_path)
    
    # Check if images were saved
    assert (tmp_path / "car").exists()
    assert (tmp_path / "truck").exists()
    assert len(list((tmp_path / "car").glob("*.jpg"))) > 0