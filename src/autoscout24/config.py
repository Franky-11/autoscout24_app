from pathlib import Path


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not determine project root.")


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ASSETS_DIR = PROJECT_ROOT / "assets"
IMAGE_DIR = ASSETS_DIR / "images"

RAW_DATA_PATH = RAW_DATA_DIR / "autoscout24.csv"
CAR_DEALER_IMAGE_PATH = IMAGE_DIR / "car_dealer.png"
