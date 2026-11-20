"""Settings for the pilot. Paths can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DERM7PT_DIR = Path(os.environ.get("DERM7PT_DIR", ROOT / "data" / "derm7pt"))
PAD_DIR = Path(os.environ.get("PAD_DIR", ROOT / "data" / "pad-ufes-20"))
WORK_DIR = Path(os.environ.get("WORK_DIR", ROOT / "work"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

SEED = 42
MAX_SIDE = 896  # images are downscaled to this before anything else

# how many images to sample per class (None = all available)
DERM7PT_PER_CLASS = {
    "melanoma": 60,
    "nevus": 60,
    "basal cell carcinoma": None,
    "seborrheic keratosis": None,
    "dermatofibroma": None,
    "vascular lesion": None,
}
PAD_PER_CLASS = 35  # PAD-UFES-20 has six classes

# answer options shown to the models; the last one lets a model decline
CLASSES = [
    "melanoma",
    "nevus",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "actinic keratosis",
    "seborrheic keratosis",
    "dermatofibroma",
    "vascular lesion",
]
OPTION_TEXT = {
    "melanoma": "Melanoma",
    "nevus": "Melanocytic nevus (mole)",
    "basal cell carcinoma": "Basal cell carcinoma",
    "squamous cell carcinoma": "Squamous cell carcinoma",
    "actinic keratosis": "Actinic keratosis",
    "seborrheic keratosis": "Seborrheic keratosis",
    "dermatofibroma": "Dermatofibroma",
    "vascular lesion": "Vascular lesion",
}
CANNOT_TELL = "Cannot be determined from this image"
LETTERS = "ABCDEFGHI"

# the seven criteria of the 7-point checklist, as asked to the models
CRITERIA = {
    "atypical_network": "an atypical pigment network",
    "blue_white_veil": "a blue-whitish veil",
    "atypical_vascular": "an atypical vascular pattern (dotted or linear irregular vessels)",
    "irregular_streaks": "irregular streaks",
    "irregular_pigmentation": "irregular pigmentation (blotches)",
    "irregular_dots": "irregular dots or globules",
    "regression": "regression structures (white scar-like or blue-grey areas)",
}

# image variants; the lesion pixels are untouched in every bg_* variant
VARIANTS = ["original", "lesion_removed", "bg_tone", "bg_marker", "bg_ruler"]
MASK_DILATE_PX = 12  # safety margin around the lesion mask

MODELS = {
    "medgemma": "google/medgemma-1.5-4b-it",
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
}

ECE_BINS = 10
N_BOOT = 2000
