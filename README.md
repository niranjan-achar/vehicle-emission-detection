# Smoky vehicle detection using machine learning

A compact web app for detecting smoke-emitting vehicles from images and videos using a paper-based machine learning pipeline with SVM classification.

## Tech Stack

- Backend: FastAPI, OpenCV, SVM-based detection pipeline
- Frontend: React
- Storage: MongoDB or JSON fallback

## What It Does

- Accepts image and video uploads
- Detects smoky vehicles using background subtraction, rear detection, feature extraction, and SVM voting
- Saves processed results and detection history
- Supports webhook and email notifications

## Run It

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
cd frontend
npm install
npm run dev
```

## Model Files

- Trained SVM bundle: `backend/weights/paper_svm_models.pkl`
- If the bundle is missing, the backend uses a fallback heuristic until training is done.

## Train in Google Colab

Use [notebooks/google_colab_paper_svm_training.ipynb](notebooks/google_colab_paper_svm_training.ipynb) to train in Colab and save the model bundle to Google Drive, then copy it into `backend/weights/`.
