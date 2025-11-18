
# Converts Zenodo Im2LaTeX dataset into trainable JSON format.

import json
from pathlib import Path
from PIL import Image
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = ROOT / "datasets" / "im2latex_raw"
OUTPUT_PATH = ROOT / "datasets" / "im2latex_prepared.json"

FORMULA_FILE = DATASET_ROOT / "im2latex_formulas.lst"
TRAIN_LST = DATASET_ROOT / "im2latex_train.lst"
VAL_LST = DATASET_ROOT / "im2latex_validate.lst"
TEST_LST = DATASET_ROOT / "im2latex_test.lst"
IMG_ROOT = DATASET_ROOT / "formula_images"


def load_formulas():
    with open(FORMULA_FILE, "r", encoding="utf-8") as f:
        formulas = [line.strip() for line in f]
    print(f"Loaded {len(formulas)} formulas")
    return formulas


def load_split(lst_path, formulas):
    samples = []
    with open(lst_path, "r") as f:
        for line in f:
            f_id, img_name, _ = line.strip().split()
            f_id = int(f_id)
            img_path = IMG_ROOT / (img_name + ".png")
            if img_path.exists():
                samples.append({
                    "img_path": str(img_path),
                    "latex": formulas[f_id]
                })
    print(f"Loaded {len(samples)} samples from {lst_path.name}")
    return samples


def main():
    if not DATASET_ROOT.exists():
        print("Dataset folder not found. Expected:", DATASET_ROOT)
        sys.exit(1)

    formulas = load_formulas()

    dataset = {
        "train": load_split(TRAIN_LST, formulas),
        "val": load_split(VAL_LST, formulas),
        "test": load_split(TEST_LST, formulas),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("\nSaved prepared dataset to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
