# Vehicle Emission Detection System (YOLOv8 + FastAPI + React)

End-to-end project to detect smoke-emitting vehicles from images/videos using YOLOv8, expose inference through FastAPI, and visualize results in a React dashboard.

## Architecture

- Backend: FastAPI + Ultralytics YOLOv8 + OpenCV
- Frontend: React (Vite) + Tailwind CSS + Axios
- Storage:
  - Primary (optional): MongoDB
  - Fallback: JSON file at `backend/storage/detections.json`

## Project Structure

```text
backend/
  app/
    main.py
    config.py
    routes/
      detect.py
    services/
      yolo_service.py
      storage_service.py
    utils/
      video_processing.py
    models/
      schema.py
  weights/
    best.pt
  requirements.txt

frontend/
  src/
    components/
      Upload.jsx
      Results.jsx
      Dashboard.jsx
    pages/
      Home.jsx
    services/
      api.js
    App.jsx
    main.jsx
    styles.css
```

## Backend API

- `GET /health`
  - Returns API and model status.
- `POST /detect/image`
  - Input: image file + optional `confidence` query param.
  - Output: detections + processed image (base64 and static path).
- `POST /detect/video`
  - Input: video file + optional `confidence` query param.
  - Output: processed video path + total detections + detection timestamps.
- `GET /detect/summary`
  - Output: dashboard counters (uploads, detections, media split).

## Setup and Run

### 1) Backend Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Place your trained YOLOv8 weights file at:

- `backend/weights/best.pt`

Then start backend:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URLs:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### 2) Frontend Setup

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Frontend URL:

- `http://localhost:5173`

## Notes

- If `backend/weights/best.pt` is missing or invalid, backend still starts in degraded mode and detection routes return `503`.
- Processed media is served from `/static/processed/...`.
- Upload file size and confidence defaults are configurable via backend environment variables.
- CORS origins are configurable through `CORS_ORIGINS` in backend `.env`.
