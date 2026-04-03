from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(
        self,
        json_db_path: Path,
        mongo_uri: str | None,
        mongo_db_name: str,
        mongo_collection_name: str,
    ) -> None:
        self.json_db_path = Path(json_db_path)
        self._lock = Lock()
        self.collection = None

        if mongo_uri:
            try:
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
                client.admin.command("ping")
                self.collection = client[mongo_db_name][mongo_collection_name]
                logger.info("Connected to MongoDB collection %s", mongo_collection_name)
            except PyMongoError as exc:
                logger.warning("MongoDB unavailable, using JSON fallback: %s", exc)

        self._ensure_json_store()

    def save_detection_record(self, record: Dict[str, Any]) -> None:
        record["created_at"] = datetime.now(timezone.utc).isoformat()

        if self.collection is not None:
            try:
                self.collection.insert_one(record)
                return
            except PyMongoError as exc:
                logger.warning("MongoDB write failed, falling back to JSON: %s", exc)

        self._append_json_record(record)

    def get_summary(self) -> Dict[str, Any]:
        records = self._read_all_records()
        total_uploads = len(records)
        total_polluting_detections = sum(int(item.get("detections_count", 0)) for item in records)
        image_uploads = sum(1 for item in records if item.get("media_type") == "image")
        video_uploads = sum(1 for item in records if item.get("media_type") == "video")

        last_updated = None
        if records:
            last_updated = records[-1].get("created_at")

        return {
            "total_uploads": total_uploads,
            "total_polluting_detections": total_polluting_detections,
            "image_uploads": image_uploads,
            "video_uploads": video_uploads,
            "last_updated": last_updated,
        }

    def _ensure_json_store(self) -> None:
        self.json_db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.json_db_path.exists():
            self.json_db_path.write_text("[]", encoding="utf-8")

    def _append_json_record(self, record: Dict[str, Any]) -> None:
        with self._lock:
            raw = self.json_db_path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw else []
            data.append(record)
            self.json_db_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read_all_records(self) -> list[Dict[str, Any]]:
        if self.collection is not None:
            try:
                docs = list(self.collection.find({}, {"_id": 0}).sort("created_at", 1))
                return docs
            except PyMongoError as exc:
                logger.warning("MongoDB read failed, using JSON fallback: %s", exc)

        return self._read_json_file()

    def _read_json_file(self) -> list[Dict[str, Any]]:
        with self._lock:
            raw = self.json_db_path.read_text(encoding="utf-8").strip()
            if not raw:
                return []
            return json.loads(raw)
