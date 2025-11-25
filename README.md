# InstaTexScanner

InstaTexScanner is a tool that takes a photo of a document (printed, handwritten, or mixed with images) and automatically converts it into **LaTeX code**. Images in the document are preserved and embedded directly into the output.

---

## 📋 Table of Contents
- [🚀 Project Goal](#-project-goal)
- [👥 Team](#-team)
- [🚀 Deployment](#-deployment)
- [📌 Features](#-features-planned)
- [📁 Project Structure](#-project-structure)
- [📂 Datasets](#-datasets)
- [⚙️ How It Works](#-how-it-works)

---

## 🚀 Project Goal
The goal is to make LaTeX document creation faster and easier by automating the process of converting handwritten notes, printed documents, or even blackboard captures into structured LaTeX.

---

## 👥 Team
Petr Belayev p.belayev@innopolis.university
Andrey Krasnov 
a.krasnov@innopolis.university 
Askar Kadyrgulov a.kadyrgulov@innopolis.university

---

Here's the updated deployment section for your README.md file with the specific links:

---

## 🚀 Deployment

### Prerequisites
- Docker
- Docker Compose

### Quick Start
1. Navigate to the deployment directory:
   ```bash
   cd code/deployment
   ```

2. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. The services will be available at:
   - **Web Application**: http://localhost:3000
   - **API Server**: http://localhost:8000/docs#/

### Stopping the Services
To stop the deployed services:
```bash
docker-compose down
```

---

The web application will be accessible at `http://localhost:3000` and the API server at `http://localhost:8000/docs#/`. Users can access the main interface through the web application URL.

---

## 📌 Features (planned)
- OCR for printed and handwritten text.
- Handwritten/printed equation recognition.
- Automatic LaTeX generation.
- Image detection and embedding in LaTeX.
- Mobile-first workflow (capture → convert → preview).
- Companion desktop/web app for editing and refinement.

---

## 📁 Project Structure

```yaml
├── code
│   ├── datasets
│   ├── deployment
│   │   ├── api
│   │   └── app
│   └── models
├── data
└── models
```

---

## 📂 Datasets
We plan to use:
- **CROHME dataset** – Handwritten mathematical expressions.
- **IAM dataset** – Handwritten English text.
- **Self-collected dataset** – Photos of handwritten lecture notes.

---

## ⚙️ How It Works
1. **Capture**: The user takes a photo of a handwritten or printed document using their phone.
2. **Preprocessing**: The app cleans the image (cropping, binarization, noise removal).
3. **Recognition**: Deep learning models process text and equations separately, converting them into LaTeX code.
4. **Image Handling**: Any detected diagrams or figures are saved and embedded into the LaTeX output as `\includegraphics`.
5. **Output**: The user receives a LaTeX file that can be compiled directly into a PDF.

---
