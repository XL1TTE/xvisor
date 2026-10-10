from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

from .types import SessionConfig, SessionMetadata


class SessionManager:
    """Manages training sessions and artifact isolation inside the .xvisor directory."""

    def __init__(self, workspace_root: Path = Path(".xvisor")) -> None:
        self.workspace_root = workspace_root
        self.sessions_dir = self.workspace_root / "sessions"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _get_meta_path(self, session_id: str) -> Path:
        return self._get_session_dir(session_id) / "metadata.json"

    def create_session(self, config: SessionConfig) -> SessionMetadata:
        session_id = str(uuid.uuid4())

        session_dir = self._get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        (session_dir / "checkpoints").mkdir(exist_ok=True)
        (session_dir / "exports").mkdir(exist_ok=True)

        metadata = SessionMetadata(
            session_id=session_id,
            name=config.name,
            architecture=config.architecture,
            num_classes=config.num_classes,
            created_at=datetime.now().isoformat(),
            current_epoch=0,
            total_epochs=0,
            last_loss=None,
            best_loss=None,
            status="created",
        )

        self._save_metadata(metadata)
        return metadata

    def _save_metadata(self, metadata: SessionMetadata) -> None:
        meta_path = self._get_meta_path(metadata.session_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

    def get_session(self, session_id: str) -> SessionMetadata:
        meta_path = self._get_meta_path(session_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"Session '{session_id}' does not exist in {self.sessions_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SessionMetadata(**data)

    def list_sessions(self) -> list[SessionMetadata]:
        if not self.sessions_dir.exists():
            return []

        sessions: list[SessionMetadata] = []
        for s_dir in sorted(self.sessions_dir.iterdir()):
            if s_dir.is_dir():
                meta_file = s_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            sessions.append(SessionMetadata(**json.load(f)))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return sessions

    def update_session(
        self,
        session_id: str,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        last_loss: Optional[float] = None,
        best_loss: Optional[float] = None,
        status: Optional[str] = None,
    ) -> SessionMetadata:
        current = self.get_session(session_id)

        updated = SessionMetadata(
            session_id=current.session_id,
            name=current.name,
            architecture=current.architecture,
            num_classes=current.num_classes,
            created_at=current.created_at,
            current_epoch=current_epoch if current_epoch is not None else current.current_epoch,
            total_epochs=total_epochs if total_epochs is not None else current.total_epochs,
            last_loss=last_loss if last_loss is not None else current.last_loss,
            best_loss=best_loss if best_loss is not None else current.best_loss,
            status=status if status is not None else current.status,
        )

        self._save_metadata(updated)
        return updated

    def get_checkpoint_dir(self, session_id: str) -> Path:
        ckpt_dir = self._get_session_dir(session_id) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def get_export_dir(self, session_id: str) -> Path:
        export_dir = self._get_session_dir(session_id) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def get_latest_checkpoint(self, session_id: str) -> Optional[Path]:
        ckpt_dir = self.get_checkpoint_dir(session_id)
        ckpts = sorted(
            ckpt_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
        return ckpts[-1] if ckpts else None
