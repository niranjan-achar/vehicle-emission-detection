from __future__ import annotations

import asyncio
import json
import logging
import smtplib
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage_dir = self.settings.json_db_path.parent
        self.file_path = self.storage_dir / "notifications.json"
        self.history_path = self.storage_dir / "notification_history.json"
        self._ensure_history()
        self._queue: "asyncio.Queue[dict]" = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._ensure_file()
        self._load()

    def _ensure_history(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self.history_path.write_text(json.dumps([]))

    def _ensure_file(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text(json.dumps(self._default_settings()))

    def _default_settings(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "min_detections": 1,
            "webhook_url": None,
            "email": {
                "enabled": False,
                "smtp_host": None,
                "smtp_port": 587,
                "smtp_user": None,
                "smtp_password": None,
                "from_email": None,
                "to_emails": "",
            },
        }

    def _load(self) -> None:
        try:
            self._data = json.loads(self.file_path.read_text())
        except Exception:
            logger.exception("Failed to load notification settings, using defaults")
            self._data = self._default_settings()

    def get_settings(self) -> dict[str, Any]:
        return self._data.copy()

    def save_settings(self, new: dict[str, Any]) -> dict[str, Any]:
        self._data.update(new)
        self.file_path.write_text(json.dumps(self._data, indent=2))
        return self.get_settings()

    def notify_async(self, record: dict[str, Any]) -> None:
        """Queue a notification job to be processed by the background worker."""
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(lambda: self._queue.put_nowait(record))
        except RuntimeError:
            # Not running in an event loop; start a temporary loop to process single job
            asyncio.run(self._queue.put(record))

    async def run_worker(self) -> None:
        """Background worker that processes queued notification jobs with retry/backoff."""
        if self._worker_task:
            return
        self._worker_task = asyncio.current_task()
        logger.info("Notification worker started")
        while True:
            job = await self._queue.get()
            await self._process_job(job)
            self._queue.task_done()

    async def _notify(self, record: dict[str, Any]) -> None:
        # legacy direct notify (not used when queueing) kept for compatibility
        if not self._data.get("enabled"):
            return

        min_det = int(self._data.get("min_detections", 1) or 1)
        if record.get("detections_count", 0) < min_det:
            return

        webhook = self._data.get("webhook_url")
        email_cfg = self._data.get("email", {}) or {}

        # Attempt webhook then email synchronously in executor
        if webhook:
            await asyncio.get_running_loop().run_in_executor(None, self._send_webhook, webhook, record)

        if email_cfg.get("enabled"):
            await asyncio.get_running_loop().run_in_executor(None, self._send_email, email_cfg, record)

    async def _process_job(self, record: dict[str, Any]) -> None:
        cfg = self.get_settings()
        if not cfg.get("enabled"):
            return
        min_det = int(cfg.get("min_detections", 1) or 1)
        if record.get("detections_count", 0) < min_det:
            return

        max_attempts = 3
        attempt = 0
        last_error = None
        while attempt < max_attempts:
            attempt += 1
            try:
                webhook = cfg.get("webhook_url")
                if webhook:
                    await asyncio.get_running_loop().run_in_executor(None, self._send_webhook, webhook, record)

                email_cfg = cfg.get("email", {}) or {}
                if email_cfg.get("enabled"):
                    await asyncio.get_running_loop().run_in_executor(None, self._send_email, email_cfg, record)

                # success
                self._append_history({"record": record, "status": "sent", "attempts": attempt})
                return
            except Exception as exc:
                last_error = str(exc)
                logger.exception("Notification attempt %s failed", attempt)
                backoff = 2 ** attempt
                await asyncio.sleep(backoff)

        # failed after retries
        self._append_history({"record": record, "status": "failed", "attempts": attempt, "error": last_error})

    def _send_webhook(self, url: str, payload: dict[str, Any]) -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info("Webhook sent, status=%s", resp.status)
        except Exception:
            logger.exception("Failed sending webhook to %s", url)

    def _append_history(self, entry: dict[str, Any]) -> None:
        try:
            content = []
            try:
                content = json.loads(self.history_path.read_text())
            except Exception:
                content = []
            content.append(entry)
            # keep only last 200 entries
            content = content[-200:]
            self.history_path.write_text(json.dumps(content, indent=2))
        except Exception:
            logger.exception("Failed to write notification history")

    def _send_email(self, cfg: dict[str, Any], payload: dict[str, Any]) -> None:
        host = cfg.get("smtp_host")
        port = int(cfg.get("smtp_port") or 587)
        user = cfg.get("smtp_user")
        password = cfg.get("smtp_password")
        from_addr = cfg.get("from_email")
        to_addrs = [s.strip() for s in (cfg.get("to_emails") or "").split(",") if s.strip()]

        if not (host and from_addr and to_addrs):
            logger.warning("Email notification not sent - SMTP config incomplete")
            return

        msg = EmailMessage()
        subject = f"Emission Detection Alert: {payload.get('media_type')}"
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        body = f"Detections: {payload.get('detections_count')}\nFile: {payload.get('file_name')}\nConfidence threshold: {payload.get('confidence_threshold')}"
        msg.set_content(body)

        try:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(msg)
                logger.info("Email notification sent to %s", to_addrs)
        except Exception:
            logger.exception("Failed to send email notification")
