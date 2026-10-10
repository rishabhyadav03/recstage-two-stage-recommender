"""Download and extract MovieLens dataset."""
import io
import os
from pathlib import Path
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def download_dataset(dataset_name: str = "ml-1m", raw_dir: str = "data/raw") -> Path:
    """Download and extract dataset if not already present."""
    if dataset_name not in URLS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {list(URLS.keys())}")

    target_dir = Path(raw_dir) / dataset_name
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Dataset '{dataset_name}' already exists at {target_dir}")
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    url = URLS[dataset_name]
    logger.info(f"Downloading {dataset_name} from {url}...")

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        content = response.read()

    logger.info(f"Extracting zip to {target_dir.parent}...")
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        z.extractall(target_dir.parent)

    logger.info(f"Dataset successfully extracted to {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_dataset("ml-1m")
