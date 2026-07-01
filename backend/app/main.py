from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.models.schema import HealthResponse
from app.routes.detect import router as detect_router
from app.routes.notifications import router as notifications_router
from app.services.notification_service import NotificationService
from app.services.paper_service import PaperService
from app.services.storage_service import StorageService
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


settings = get_settings()

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    app.state.paper_service = PaperService(config=settings.dict())
    app.state.storage_service = StorageService(
        json_db_path=settings.json_db_path,
        mongo_uri=settings.mongo_uri,
        mongo_db_name=settings.mongo_db_name,
        mongo_collection_name=settings.mongo_collection_name,
    )

    # initialize notification service (reads/creates notifications.json)
    app.state.notification_service = NotificationService()
    # start background worker for notifications
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(app.state.notification_service.run_worker())
    except Exception:
        logger.exception("Failed to start notification worker")

    try:
        app.state.paper_service.load()
        app.state.model_loaded = True
        app.state.model_classes = ["smoky", "non_smoky"]
        app.state.smoke_class_ids = [0]  # smoky class

        logger.info("Paper-based detection service loaded successfully")
    except Exception as exc:
        app.state.model_loaded = False
        app.state.model_classes = []
        app.state.smoke_class_ids = []
        logger.exception("Failed to load paper service: %s", exc)
        logger.info("Detection endpoints will return 503 until service is fixed.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if app.state.model_loaded else "degraded",
        model_loaded=bool(app.state.model_loaded),
        model_path="paper-based-detection",
        environment=settings.app_env,
        smoke_class_names=settings.smoke_class_names_list,
        resolved_smoke_class_ids=app.state.smoke_class_ids,
        model_classes=app.state.model_classes,
        warning="Using paper-based detection (IET 2019) - no YOLOv8 model loaded",
    )


app.include_router(detect_router)
app.include_router(notifications_router)
app.mount("/static", StaticFiles(directory=settings.processed_dir.parent), name="static")
