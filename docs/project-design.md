# Project Design: Human Detection & ReID Model Forge (`xvisor`)

## 1. System Overview

`xvisor` is an engineering platform and MLOps CLI tool designed to **train, evaluate, and export** computer vision models for human detection and visual re-identification (ReID).

The primary purpose of this tool is to act as a **Model Forge**:
- It produces production-ready **`.onnx` neural network artifacts** (`human_detector.onnx` and `person_reid.onnx`).
- Downstream applications (such as a Vue/Node.js web app, C++, or C# applications) consume these exported models directly via ONNX Runtime to perform live multi-object tracking and video inference.
- Model lifecycle management is handled via **Sessions** isolated inside a dedicated `.xvisor/` workspace.
- The CLI provides human-readable output as well as a `--json` IPC streaming mode for integration with external host processes.

---

## 2. System Architecture

The architecture is organized into decoupled layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Interface & IPC Layer                      │
│            [Typer CLI]    [--json Structured IPC Stream]               │
│               session | train | test | export                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Session & MLOps Layer                           │
│  - SessionManager: Manages isolated workspace under .xvisor/           │
│  - Storage Hierarchy:                                                  │
│      .xvisor/sessions/<session_id>/                                    │
│        ├── metadata.json       (hyperparameters, status, metrics)      │
│        ├── checkpoints/        (full training session .pth files)      │
│        └── exports/            (exported .onnx models)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Execution Engine Layer                          │
│  ├── Trainer             : Forward/backward loop, device management    │
│  ├── CheckpointManager   : Full session persistence & resumption       │
│  ├── DetectorEvaluator   : Frame-level mAP calculation (50 & 50:95)    │
│  └── ModelExporter       : PyTorch -> ONNX (dynamic batch/spatial axes)│
└───────────────┬───────────────────────────────────────┬────────────────┘
                │                                       │
                ▼                                       ▼
┌───────────────────────────────┐       ┌────────────────────────────────┐
│         Data Pipeline         │       │          Model Layer           │
│  ├── SourceDetector (local)   │       │  ├── Detector (Faster R-CNN)   │
│  ├── LocalFetcher (verify)    │       │  │   - MobileNetV3 / ResNet50  │
│  ├── MediaClassifier          │       │  │   - Custom 2-class head     │
│  ├── FrameProvider (lazy)     │       │  └── ReIDModel (MobileNetV3)   │
│  │   (dir, video, zip)        │       │      - L2-normalized 576D      │
│  ├── AnnotationParser         │       │        embedding vectors       │
│  │   (COCO JSON, MOT TXT)     │       └────────────────────────────────┘
│  └── TrackingDataset (PyTorch)│
└───────────────────────────────┘
```

---

## 3. Subsystem Specifications

### 3.1 Session Management Subsystem (`src/session/`)
- Encapsulates every training experiment inside `.xvisor/sessions/<session_id>/`.
- Tracks session states (`created`, `training`, `completed`, `resuming`), epochs, loss metrics, and architecture settings.
- Automatically discovers the latest checkpoint when resuming.

### 3.2 Model Subsystem (`src/detector/`, `src/reid/`)
- **Faster R-CNN Detector**: Two-stage detector with custom 2-class box predictor (Class 0: background, Class 1: human). Supports MobileNetV3 and ResNet50 backbones.
- **ReID Feature Extractor**: MobileNetV3-based convolutional embedder taking image patches $[B, 3, 128, 64]$ and outputting L2-normalized 576-dimensional embedding vectors for person appearance discrimination.

### 3.3 Data Pipeline Subsystem (`src/data/`)
- **Source Detection**: Matches input strings into strongly-typed `LocalFile` or `LocalDirectory` entities.
- **Media Classification & Frame Access**: Classifies inputs into `ImageDirectoryMedia`, `VideoFileMedia`, or `ArchiveMedia`. Provides a lazy `FrameProvider` (`get_frame(index) -> PIL Image`) to keep memory footprint under a few megabytes regardless of video size.
- **Annotation Parsing**: Parses COCO `.json` and MOT `.txt` ground-truth files into normalized `[x1, y1, x2, y2]` coordinates.
- **PyTorch Integration**: `TrackingDataset` and `collate_tracking_batch` format variable-length targets into PyTorch batches.

### 3.4 Execution & Evaluation Subsystem (`src/engine/`)
- **Trainer**: Manages device transfer (CUDA/CPU), forward pass, multi-loss accumulation, backpropagation, and learning rate scheduling. Emits decoupled `EpochProgress` events.
- **CheckpointManager**: Serializes model weights, optimizer momentum, scheduler states, and metadata for exact resumption.
- **DetectorEvaluator**: Computes $AP_{50}$ and $mAP_{50:95}$ across test sets using standard 11-point interpolated Precision-Recall curves.

### 3.5 Export Subsystem (`src/export/`)
- **`export_detector_to_onnx`**: Serializes Faster R-CNN with dynamic spatial axes (`[1, 3, height, width]`).
- **`export_reid_to_onnx`**: Serializes ReID embedder with dynamic batch axis (`[batch_size, 3, 128, 64] -> [batch_size, 576]`).
- **`verify_onnx_model`**: Runs graph validation via `onnx.checker`.

---

## 4. CLI & IPC Interface (`main.py`)

All commands support human-readable terminal output and `--json` streaming:

```bash
# Session Management
xvisor session --name "my-experiment" --arch mobilenet_v3
xvisor session --list [--json]
xvisor session --resume <session_id>

# Training & Resumption
xvisor train --session <id> --data <path> --annotations <path> --epochs 10 [--json]
xvisor train --session <id> --data <path> --annotations <path> --epochs 5 --resume [--json]

# Accuracy Evaluation (mAP)
xvisor test --session <id> --data <path> --annotations <path> [--json]

# Export to ONNX
xvisor export --session <id> [-o custom/output/dir] [--json]
```

---

## 5. Automated CI Testing (`.github/workflows/ci.yml`)

- **`Tests` (`ci.yml`)**: Automated Linux test suite running on `ubuntu-latest` on every push to `main` and on pull requests.
- Employs Poetry virtualenv caching to keep execution times under 30 seconds.
