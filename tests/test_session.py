from pathlib import Path
import pytest

from session import SessionManager, SessionConfig


def test_session_creation(tmp_path: Path) -> None:
    manager = SessionManager(workspace_root=tmp_path / ".xvisor")
    meta = manager.create_session(
        SessionConfig(
            architecture="mobilenet_v3",
            num_classes=2,
            name="test_experiment",
        )
    )

    assert meta.session_id.startswith("sess_")
    assert meta.name == "test_experiment"
    assert meta.architecture == "mobilenet_v3"
    assert meta.status == "created"

    # Verify directory structure
    session_dir = tmp_path / ".xvisor" / "sessions" / meta.session_id
    assert session_dir.exists()
    assert (session_dir / "checkpoints").exists()
    assert (session_dir / "exports").exists()
    assert (session_dir / "metadata.json").exists()


def test_session_list_and_get(tmp_path: Path) -> None:
    manager = SessionManager(workspace_root=tmp_path / ".xvisor")
    s1 = manager.create_session(SessionConfig(architecture="mobilenet_v3", name="s1"))
    s2 = manager.create_session(SessionConfig(architecture="resnet_50", name="s2"))

    all_sessions = manager.list_sessions()
    assert len(all_sessions) == 2
    ids = {s.session_id for s in all_sessions}
    assert s1.session_id in ids
    assert s2.session_id in ids

    retrieved = manager.get_session(s1.session_id)
    assert retrieved.name == "s1"


def test_session_update_and_latest_checkpoint(tmp_path: Path) -> None:
    manager = SessionManager(workspace_root=tmp_path / ".xvisor")
    meta = manager.create_session(SessionConfig(architecture="mobilenet_v3"))

    # Create dummy checkpoints
    ckpt_dir = manager.get_checkpoint_dir(meta.session_id)
    (ckpt_dir / "checkpoint_epoch_1.pth").touch()
    (ckpt_dir / "checkpoint_epoch_2.pth").touch()

    latest = manager.get_latest_checkpoint(meta.session_id)
    assert latest is not None
    assert latest.name == "checkpoint_epoch_2.pth"

    updated = manager.update_session(
        meta.session_id,
        current_epoch=2,
        total_epochs=10,
        last_loss=0.35,
        status="training",
    )
    assert updated.current_epoch == 2
    assert updated.last_loss == 0.35
    assert updated.status == "training"
