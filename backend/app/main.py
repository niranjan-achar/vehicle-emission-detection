from __future__ import annotations

import logging

from app.config import get_settings
from app.models.schema import HealthResponse
from app.routes.detect import router as detect_router
from app.services.storage_service import StorageService
from app.services.yolo_service import YoloService
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

    app.state.yolo_service = YoloService(model_path=settings.model_path)
    app.state.storage_service = StorageService(
        json_db_path=settings.json_db_path,
        mongo_uri=settings.mongo_uri,
        mongo_db_name=settings.mongo_db_name,
        mongo_collection_name=settings.mongo_collection_name,
    )

    try:
        fallback_name = settings.fallback_model_name if settings.allow_fallback_model else None
        app.state.yolo_service.load(fallback_model_name=fallback_name)
        app.state.model_loaded = True
        app.state.model_classes = app.state.yolo_service.get_available_classes()
        app.state.smoke_class_ids = app.state.yolo_service.resolve_class_ids(
            settings.smoke_class_names_list
        )

        if not app.state.smoke_class_ids:
            warning = (
                "Configured smoke classes were not found in model labels. "
                f"Configured={settings.smoke_class_names_list}, Available={app.state.model_classes}. "
                "Use a smoke-trained model or update SMOKE_CLASS_NAMES in backend/.env."
            )
            if app.state.yolo_service.model_warning:
                app.state.yolo_service.model_warning = f"{app.state.yolo_service.model_warning} | {warning}"
            else:
                app.state.yolo_service.model_warning = warning
    except Exception as exc:
        app.state.model_loaded = False
        app.state.model_classes = []
        app.state.smoke_class_ids = []
        logger.exception("Failed to load YOLO model from %s", settings.model_path)
        logger.info("Detection endpoints will return 503 until model is fixed. Error: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if app.state.model_loaded else "degraded",
        model_loaded=bool(app.state.model_loaded),
        model_path=str(app.state.yolo_service.model_path),
        environment=settings.app_env,
        smoke_class_names=settings.smoke_class_names_list,
        resolved_smoke_class_ids=app.state.smoke_class_ids,
        model_classes=app.state.model_classes,
        warning=app.state.yolo_service.model_warning,
    )


app.include_router(detect_router)
app.mount("/static", StaticFiles(directory=settings.processed_dir.parent), name="static")
