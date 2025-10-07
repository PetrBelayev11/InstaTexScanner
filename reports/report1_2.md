# InstaTexScanner — Project Progress Report

## Team Information
**Team Name:** InstaTexScanner  
**Participants:**  
- Petr Belayev (p.belayev@innopolis.university)  
- Andrey Krasnov (a.krasnov@innopolis.university)  
- Askar Kadyrgulov (a.kadyrgulov@innopolis.university)  

## Repository Link
**GitHub Repository:** https://github.com/petrbelayev11/InstaTexScanner  

## Project Topic and Description
The project aims to develop InstaTexScanner — a software tool that takes photos of handwritten or printed documents and automatically converts them into LaTeX code, preserving both text and embedded images. The solution is designed to simplify the digitization of lecture notes, mathematical formulas, and research papers. This tool is valuable for students, researchers, and educators who work extensively with LaTeX and want to avoid manually retyping complex expressions.

## What We Have Done So Far
We created the initial prototype demonstrating data generation, preprocessing, and model training workflow. The implementation includes three major components:  
1. **data_and_model_demo.py** — generates synthetic datasets simulating printed and handwritten text using Pillow, applies distortions to mimic handwriting, preprocesses images, and optionally trains a simple CNN model using EMNIST.  
2. **preprocess.py** — performs image binarization and thresholding for OCR readiness.  
3. **model_stub.py** — placeholder structure for the future im2markup-style neural model that will convert image features to LaTeX code.

## Explanation of Code and Design Choices
The architecture was designed with modularity and extensibility in mind. The demo script separates three essential phases: data generation, preprocessing, and model training. Pillow was used to generate both printed and distorted handwritten-like text to allow model testing without requiring a large handwritten dataset early on. Preprocessing includes grayscale conversion and global binarization, preparing the input for OCR and neural models. The CNN implemented in PyTorch serves as a lightweight test model to validate the data pipeline and ensure future deep models (like Transformer-based LaTeX generators) can be integrated easily.

## Intermediate Results
The demo successfully generates synthetic samples that resemble handwritten notes and processes them into binarized images. The preprocessing and dataset creation scripts were tested on Windows and Linux environments. A small CNN trained on EMNIST (if PyTorch is available) verified that the pipeline runs end-to-end, from data preparation to model training. Example outputs and binarized images are included in the package to illustrate current progress.

## Future Work and Plans
Next steps include:  
1. **Dataset Expansion** — Integrate CROHME (handwritten math) and IAM (handwritten text) datasets, and collect additional handwritten notes.  
2. **Model Development** — Implement the full encoder-decoder (im2markup) architecture for image-to-LaTeX translation, and test Transformer-based models such as TroCR for OCR tasks.  
3. **Backend Integration** — Develop a server-side API for inference and a mobile app for photo capture and uploading.  
4. **Evaluation** — Measure model performance using metrics such as Word Error Rate (WER), expression-level accuracy, and LaTeX compilability.  
5. **Optimization** — Explore quantization for mobile devices and improve inference speed and reliability.

## Success Criteria
The project will be considered successful if InstaTexScanner can accurately convert images of documents into compilable LaTeX with over 85% end-to-end success rate and achieve 90% recognition accuracy on handwritten and printed text datasets.

## Team Deliverables
- **Petr:** Data and model prototyping, report writing.  
- **Askar:** Assembling the GitHub pipeline, API code prototype, data and model prototyping.  
- **Andrey:** Debugging.
