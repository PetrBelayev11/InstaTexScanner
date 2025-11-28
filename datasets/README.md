If you want to retrain model this is a little instruction about how to setup database.

# Dataset Setup Guide

This directory is intended for the Im2LaTeX-100K dataset. Follow the instructions below to download and prepare the data for model training.

## Step 1: Download and Extract the Dataset

1.  **Download the dataset** from the following link:
    *   [Im2LaTeX-100K Download Link](https://zenodo.org/api/records/56198/files-archive)

2.  **Extract the contents** of the downloaded archive file (e.g., `im2latex-100k.zip` or `.tar.gz`) directly into this `datasets\` folder.

    After extraction, your folder structure should look similar to this:
    ```
    └── datasets/         <-- You are here
        ├── im2latex/
        │   ├── formula_images/
        │   ├── im2latex_formulas.lst
        │   ├── im2latex_test.lst
        │   ├── im2latex_train.lst
        │   └── im2latex_validate.lst
        ├── prepare_im2latex.py
        └── README.md
    ```

## Step 2: Prepare the Data for Training

Once the dataset is in place, you need to run the preparation script.

From this directory, run the following command:

```bash
python prepare_im2latex.py
```

This script will process the raw images and formulas into a format ready for training the model.
