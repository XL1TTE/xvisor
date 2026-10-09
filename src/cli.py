from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import typer
import torch

from device import DeviceProvider
from detector import Detector, DetectorConfig, DetectorArchitecture
from data.sources import SourceDetector
from data.fetching import LocalFetcher, LocalFetchSuccess
from data.media import classify_media, resolve_frame_provider
from data.annotations import classify_annotation_file, parse_annotations
from data.dataset import TrackingDataset
from engine import (
    Trainer,
    TrainingConfig,
    EpochProgress,
    DetectorEvaluator,
    EvaluationConfig,
    CheckpointManager,
)
from reid import ReIDModel
from export import export_detector_to_onnx, export_reid_to_onnx, verify_onnx_model
from session import SessionManager, SessionConfig

app = typer.Typer(
    name="xvisor",
    help="XVISOR: Human Tracking Model Forge & Training Platform",
    no_args_is_help=True,
)


def _load_dataset(data_path: Path, annotations_path: Path) -> TrackingDataset:
    detector = SourceDetector.default()
    src = detector.detect(str(data_path))

    fetched = LocalFetcher.fetch(src)
    if not isinstance(fetched, LocalFetchSuccess):
        raise ValueError(f"Failed to fetch data source: {fetched.error_message}")

    media = classify_media(fetched.resolved)
    frame_provider = resolve_frame_provider(media)

    ann_meta = classify_annotation_file(annotations_path)
    annotations = parse_annotations(ann_meta)

    return TrackingDataset(frame_provider=frame_provider, annotations=annotations)


@app.command("session")
def session_cmd(
    name: str = typer.Option("experiment", "--name", "-n", help="Name for new session"),
    arch: str = typer.Option("mobilenet_v3", "--arch", "-a", help="Architecture [mobilenet_v3 / resnet_50]"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List all sessions"),
    resume_id: Optional[str] = typer.Option(None, "--resume", "-r", help="Session ID to inspect/resume"),
    json_mode: bool = typer.Option(False, "--json", help="Output as JSON for IPC"),
) -> None:
    """Create a new session, list existing sessions, or inspect a session to resume."""
    manager = SessionManager()

    if list_sessions:
        sessions = manager.list_sessions()
        if json_mode:
            print(json.dumps({"event": "session_list", "sessions": [asdict(s) for s in sessions]}))
        else:
            if not sessions:
                typer.echo("[xvisor] No active sessions found in .xvisor/sessions/")
                return
            typer.echo(f"[xvisor] Active Sessions ({len(sessions)}):")
            typer.echo(f"{'ID':<38} {'NAME':<20} {'ARCH':<15} {'EPOCH':<8} {'STATUS':<10}")
            typer.echo("-" * 95)
            for s in sessions:
                typer.echo(f"{s.session_id:<38} {s.name:<20} {s.architecture:<15} {s.current_epoch:<8} {s.status:<10}")
        return

    if resume_id:
        meta = manager.get_session(resume_id)
        latest_ckpt = manager.get_latest_checkpoint(resume_id)
        if json_mode:
            print(
                json.dumps(
                    {
                        "event": "session_resume_ready",
                        "session": asdict(meta),
                        "latest_checkpoint": str(latest_ckpt) if latest_ckpt else None,
                    }
                )
            )
        else:
            typer.echo(f"[xvisor] Session ready for resumption: {meta.session_id} ('{meta.name}')")
            typer.echo(f"  Architecture: {meta.architecture}")
            typer.echo(f"  Current Epoch: {meta.current_epoch}")
            typer.echo(f"  Latest Checkpoint: {latest_ckpt}")
        return

    # Create new session
    cfg = SessionConfig(architecture=DetectorArchitecture(arch).value, name=name, num_classes=2)
    meta = manager.create_session(cfg)

    if json_mode:
        print(json.dumps({"event": "session_created", "session": asdict(meta)}))
    else:
        typer.echo(f"[xvisor] Created session: {meta.session_id} ('{meta.name}')")
        typer.echo(f"  Architecture: {meta.architecture}")
        typer.echo(f"  Workspace: .xvisor/sessions/{meta.session_id}")


@app.command("train")
def train_cmd(
    session: str = typer.Option(..., "--session", "-s", help="Session ID"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to images directory or video file"),
    annotations: Path = typer.Option(..., "--annotations", help="Path to COCO JSON or MOT TXT file"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Number of epochs to train"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(0.001, "--lr", help="Learning rate"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from latest checkpoint in session"),
    json_mode: bool = typer.Option(False, "--json", help="Output as JSON for IPC"),
) -> None:
    """Train or resume training a session's detector model."""
    manager = SessionManager()
    device = DeviceProvider().get_device()
    meta = manager.get_session(session)

    dataset = _load_dataset(data, annotations)
    ckpt_dir = manager.get_checkpoint_dir(meta.session_id)

    if resume:
        latest_ckpt = manager.get_latest_checkpoint(meta.session_id)
        if latest_ckpt is None:
            raise typer.BadParameter(f"No checkpoint found to resume in session: '{session}'")

        target_total_epochs = meta.current_epoch + epochs
        train_cfg = TrainingConfig(
            epochs=target_total_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            save_dir=ckpt_dir,
        )
        detector_cfg = DetectorConfig(
            architecture=DetectorArchitecture(meta.architecture),
            num_classes=meta.num_classes,
            pretrain=False,
        )
        model = Detector.create(detector_cfg)
        trainer = Trainer(model, detector_cfg, train_cfg, device)
        trainer.resume_from_checkpoint(latest_ckpt)
        manager.update_session(meta.session_id, status="resuming", total_epochs=target_total_epochs)
        if not json_mode:
            typer.echo(f"[xvisor] Resumed session {session} from epoch {meta.current_epoch}. Target: {target_total_epochs} epochs.")
    else:
        train_cfg = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            save_dir=ckpt_dir,
        )
        detector_cfg = DetectorConfig(
            architecture=DetectorArchitecture(meta.architecture),
            num_classes=meta.num_classes,
            pretrain=True,
        )
        model = Detector.create(detector_cfg)
        trainer = Trainer(model, detector_cfg, train_cfg, device)
        manager.update_session(meta.session_id, status="training", total_epochs=epochs)

    def on_progress(p: EpochProgress) -> None:
        if json_mode:
            print(
                json.dumps(
                    {
                        "event": "epoch_progress",
                        "session": meta.session_id,
                        "epoch": p.epoch + 1,
                        "total_epochs": p.total_epochs,
                        "batch": p.batch_index + 1,
                        "total_batches": p.total_batches,
                        "loss": round(p.loss, 4),
                    }
                )
            )
        else:
            typer.echo(
                f"\r[xvisor] Epoch {p.epoch + 1}/{p.total_epochs} | Batch {p.batch_index + 1}/{p.total_batches} | Loss: {p.loss:.4f}",
                nl=False,
            )

    saved_ckpts = trainer.train(dataset, on_progress=on_progress)
    if not json_mode:
        typer.echo()

    latest_loss = None
    if saved_ckpts:
        payload = CheckpointManager.load(saved_ckpts[-1], device=torch.device("cpu"))
        latest_loss = payload.metadata.loss

    new_epoch_count = train_cfg.epochs
    manager.update_session(
        meta.session_id,
        current_epoch=new_epoch_count,
        last_loss=latest_loss,
        status="completed",
    )

    if json_mode:
        print(json.dumps({"event": "training_completed", "session": meta.session_id, "checkpoints": [str(c) for c in saved_ckpts]}))
    else:
        typer.echo(f"[xvisor] Training completed. Saved {len(saved_ckpts)} checkpoints in {ckpt_dir}")


@app.command("test")
def test_cmd(
    session: str = typer.Option(..., "--session", "-s", help="Session ID"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to test images or video"),
    annotations: Path = typer.Option(..., "--annotations", help="Path to test annotations"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    conf: float = typer.Option(0.5, "--conf", help="Confidence threshold"),
    json_mode: bool = typer.Option(False, "--json", help="Output as JSON for IPC"),
) -> None:
    """Evaluate a session's detector model accuracy (mAP)."""
    manager = SessionManager()
    device = DeviceProvider().get_device()
    meta = manager.get_session(session)

    latest_ckpt = manager.get_latest_checkpoint(meta.session_id)
    if latest_ckpt is None:
        raise typer.BadParameter(f"No checkpoint found in session: '{session}'")

    dataset = _load_dataset(data, annotations)

    detector_cfg = DetectorConfig(
        architecture=DetectorArchitecture(meta.architecture),
        num_classes=meta.num_classes,
        pretrain=False,
    )
    model = Detector.create(detector_cfg)

    payload = CheckpointManager.load(latest_ckpt, device=device)
    model.load_state_dict(payload.model_state_dict)

    evaluator = DetectorEvaluator(
        model=model,
        device=device,
        config=EvaluationConfig(batch_size=batch_size, confidence_threshold=conf),
    )

    metrics = evaluator.evaluate(dataset)

    if json_mode:
        print(
            json.dumps(
                {
                    "event": "test_completed",
                    "session": meta.session_id,
                    "metrics": {
                        "ap50": round(metrics.ap50, 4),
                        "map50_95": round(metrics.map50_95, 4),
                        "precision": round(metrics.precision, 4),
                        "recall": round(metrics.recall, 4),
                        "total_ground_truth": metrics.total_ground_truth,
                        "total_predictions": metrics.total_predictions,
                    },
                }
            )
        )
    else:
        typer.echo(f"[xvisor] Evaluation Results for Session: {meta.session_id}")
        typer.echo(f"  AP50:               {metrics.ap50:.4f}")
        typer.echo(f"  mAP (50:95):        {metrics.map50_95:.4f}")
        typer.echo(f"  Precision:          {metrics.precision:.4f}")
        typer.echo(f"  Recall:             {metrics.recall:.4f}")
        typer.echo(f"  Ground Truth Count: {metrics.total_ground_truth}")
        typer.echo(f"  Predictions Count:  {metrics.total_predictions}")


@app.command("export")
def export_cmd(
    session: str = typer.Option(..., "--session", "-s", help="Session ID"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom output directory for ONNX models"),
    json_mode: bool = typer.Option(False, "--json", help="Output as JSON for IPC"),
) -> None:
    """Export detector and ReID models to ONNX for consumption in external applications."""
    manager = SessionManager()
    meta = manager.get_session(session)

    latest_ckpt = manager.get_latest_checkpoint(meta.session_id)
    if latest_ckpt is None:
        raise typer.BadParameter(f"No checkpoint found to export in session: '{session}'")

    detector_cfg = DetectorConfig(
        architecture=DetectorArchitecture(meta.architecture),
        num_classes=meta.num_classes,
        pretrain=False,
    )
    detector_model = Detector.create(detector_cfg)
    payload = CheckpointManager.load(latest_ckpt, device=torch.device("cpu"))
    detector_model.load_state_dict(payload.model_state_dict)

    target_dir = output or manager.get_export_dir(meta.session_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Detector ONNX
    det_onnx_path = target_dir / "human_detector.onnx"
    export_detector_to_onnx(detector_model, det_onnx_path)
    verify_onnx_model(det_onnx_path)

    # 2. Export ReID ONNX
    reid_model = ReIDModel(pretrained=True)
    reid_onnx_path = target_dir / "person_reid.onnx"
    export_reid_to_onnx(reid_model, reid_onnx_path)
    verify_onnx_model(reid_onnx_path)

    if json_mode:
        print(
            json.dumps(
                {
                    "event": "export_completed",
                    "session": meta.session_id,
                    "exports": {
                        "detector": str(det_onnx_path),
                        "reid": str(reid_onnx_path),
                    },
                }
            )
        )
    else:
        typer.echo(f"[xvisor] Successfully exported models for Session: {meta.session_id}")
        typer.echo(f"  Detector: {det_onnx_path}")
        typer.echo(f"  ReID:     {reid_onnx_path}")
