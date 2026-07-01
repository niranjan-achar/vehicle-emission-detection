from __future__ import annotations

import json

from app.services.notification_service import NotificationService
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/notifications", tags=["Notifications"])


class EmailConfig(BaseModel):
    enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int | None = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    from_email: str | None = None
    to_emails: str | None = ""


class NotificationSettings(BaseModel):
    enabled: bool = False
    min_detections: int = 1
    webhook_url: str | None = None
    email: EmailConfig = EmailConfig()


@router.get("", response_model=NotificationSettings)
def get_notifications(request: Request) -> NotificationSettings:
    svc: NotificationService = request.app.state.notification_service
    data = svc.get_settings()
    return NotificationSettings(**data)


@router.post("", response_model=NotificationSettings)
def update_notifications(request: Request, payload: NotificationSettings) -> NotificationSettings:
    svc: NotificationService = request.app.state.notification_service
    new = svc.save_settings(payload.dict())
    return NotificationSettings(**new)


@router.get("/history")
def get_history(request: Request):
    svc: NotificationService = request.app.state.notification_service
    try:
        raw = svc.history_path.read_text()
        data = []
        try:
            data = json.loads(raw)
        except Exception:
            data = []
        return {"history": data}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read notification history")


@router.post("/test")
def test_notification(request: Request):
    svc: NotificationService = request.app.state.notification_service
    # simple test payload
    payload = {"media_type": "test", "file_name": "test", "detections_count": 1, "confidence_threshold": 0.5}
    try:
        svc.notify_async(payload)
        return {"ok": True}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to schedule test notification")
