from pathlib import Path
import os


TRAINING_DIR = Path(__file__).parent.parent.parent / "dataset"
TESTING_DIR = (
    Path(os.environ["KITS23_TEST_DIR"]).resolve(strict=True)
    if "KITS23_TEST_DIR" in os.environ.keys() else None
)
SRC_DIR = (
    Path(os.environ["KITS23_SERVER_DATA"]).resolve(strict=True)
    if "KITS23_SERVER_DATA" in os.environ.keys() else None
)
KITS21_PATH = (
    Path(os.environ["KITS21_PATH"]).resolve(strict=True)
    if "KITS21_PATH" in os.environ.keys() else None
)
CACHE_FILE = Path(__file__).parent.parent / "annotation" / "cache.json"
