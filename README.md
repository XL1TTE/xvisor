<div align="center">

# 🎯 XVISOR

**A modular MLOps platform and CLI forge for training, evaluating, and exporting human detection and re-identification models.**

[![Tests](https://github.com/XL1TTE/xvisor/actions/workflows/ci.yml/badge.svg)](https://github.com/XL1TTE/xvisor/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/Export-ONNX-005CED?logo=onnx)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#cli-usage">CLI Usage</a> •
  <a href="#dataset-formats">Dataset Formats</a> •
  <a href="#ipc--automation-mode">IPC / Web App Mode</a> •
  <a href="#architecture-docs">Architecture Docs</a>
</p>

---

</div>

## 💡 What is XVISOR?

**XVISOR** is a specialized Computer Vision Model Forge. It provides an isolated, session-driven environment to **train, fine-tune, evaluate ($mAP$), and export** neural networks for human tracking pipelines.

Instead of locking models inside a monolithic Python application, XVISOR outputs standardized, production-grade **`.onnx` neural network artifacts**:
1. **`human_detector.onnx`**: Faster R-CNN detector extracting bounding boxes $[x_1, y_1, x_2, y_2]$ from video frames.
2. **`person_reid.onnx`**: Deep feature extractor generating 576-dimensional visual appearance vectors for person re-identification.

These exported models can be loaded directly into external web applications, desktop applications, or native backends (C++, C#, Node.js, Go, Rust) using **ONNX Runtime**.

---

## ✨ Key Features

- ⚡ **Zero-Config Hardware Selection**: Automatically detects NVIDIA CUDA GPUs for hardware acceleration and gracefully falls back to multi-threaded CPU.
- 📁 **Isolated MLOps Sessions**: Every run is encapsulated under `.xvisor/sessions/<session_id>/` with independent checkpoints, metrics, and exports.
- 🔁 **Lossless Training Resumption**: Resumes training sessions with full optimizer momentum, learning rate scheduler steps, and epoch state intact.
- 🎯 **Decoupled Accuracy Testing**: Evaluates average precision ($AP_{50}$ and $mAP_{50:95}$) on static images or video frames using standard 11-point interpolated Precision-Recall curves.
- 📦 **Multi-Format Ingestion**: Reads training imagery directly from loose image directories, raw video files (`.mp4`, `.avi`, `.mkv`), or compressed `.zip` archives.
- 🏷️ **Dual Annotation Support**: Transparently parses both **COCO JSON** and **MOTChallenge TXT** annotations.
- 🔌 **Machine-Readable IPC Mode**: All CLI commands support a `--json` flag, streaming structured JSON events over `stdout` for headless integration with desktop/web UI backends.

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/XL1TTE/xvisor.git
cd xvisor

# 2. Install dependencies
poetry install

# 3. Verify installation & view commands
poetry run xvisor --help
```

> **Tip:** You can prefix commands with `poetry run xvisor ...`, or activate your virtual environment once (`source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`) to run `xvisor` directly.

---

## 🛠️ CLI Usage

All commands use the unified `xvisor` CLI interface.

### 1. Manage Sessions
Every experiment starts with a session. Sessions isolate training checkpoints and model exports.

```bash
# Create a new session with MobileNetV3 backbone
xvisor session --name "cctv_experiment" --arch mobilenet_v3

# List all active sessions
xvisor session --list

# Inspect an existing session
xvisor session --resume 3fa85f64-5717-4562-b3fc-2c963f66afa6
```

---

### 2. Train a Detector
Train the human detector on your custom dataset. Checkpoints are automatically saved under `.xvisor/sessions/<session_id>/checkpoints/`.

```bash
xvisor train \
  --session 3fa85f64-5717-4562-b3fc-2c963f66afa6 \
  --data path/to/frames_folder_or_video.mp4 \
  --annotations path/to/annotations.json \
  --epochs 10 \
  --batch-size 2 \
  --lr 0.001
```

#### Resume an Interrupted Session
Continue training an existing session from its latest saved checkpoint:

```bash
xvisor train \
  --session 3fa85f64-5717-4562-b3fc-2c963f66afa6 \
  --data path/to/data \
  --annotations path/to/annotations.json \
  --epochs 5 \
  --resume
```

---

### 3. Evaluate Detection Error ($mAP$)
Benchmark a session's model accuracy against ground-truth labels.

```bash
xvisor test \
  --session 3fa85f64-5717-4562-b3fc-2c963f66afa6 \
  --data path/to/test_data \
  --annotations path/to/test_annotations.json \
  --conf 0.5
```

**Output:**
```text
[xvisor] Evaluation Results for Session: 3fa85f64-5717-4562-b3fc-2c963f66afa6
  AP50:               0.8842
  mAP (50:95):        0.6120
  Precision:          0.9125
  Recall:             0.8450
  Ground Truth Count: 342
  Predictions Count:  318
```

---

### 4. Export Models to ONNX
Export your trained detector and pretrained visual ReID embedder for external deployment.

```bash
# Export to default session directory (.xvisor/sessions/<id>/exports/)
xvisor export --session 3fa85f64-5717-4562-b3fc-2c963f66afa6

# Export directly to a custom destination
xvisor export --session 3fa85f64-5717-4562-b3fc-2c963f66afa6 -o ./models/
```

**Generated files:**
- `human_detector.onnx`: Dynamic spatial axes (`[1, 3, height, width]`).
- `person_reid.onnx`: Dynamic batch axis (`[batch_size, 3, 128, 64] -> [batch_size, 576]`).

---

## 📂 Dataset Formats Supported

XVISOR transparently accepts both standard computer vision formats:

### 1. COCO Format (`.json`)
Standard format from object detection benchmarks:
```json
{
  "images": [{"id": 1, "file_name": "frame_001.jpg"}],
  "categories": [{"id": 1, "name": "person"}],
  "annotations": [
    {"image_id": 1, "category_id": 1, "bbox": [100.0, 150.0, 40.0, 80.0]}
  ]
}
```

### 2. MOTChallenge Format (`.txt` / `.csv`)
Sequential video frame tracking format:
```text
<frame_id>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
1, 101, 100.0, 150.0, 40.0, 80.0, 1, 1, 1.0
1, 102, 320.0, 180.0, 50.0, 95.0, 1, 1, 1.0
2, 101, 105.0, 152.0, 40.0, 80.0, 1, 1, 1.0
```

---

## 🤖 IPC / Automation Mode (External Integration)

When wrapping XVISOR inside desktop apps, web services, or backend processes, add the `--json` flag to stream machine-readable event objects:

```bash
xvisor train --session 3fa85f64-5717-4562-b3fc-2c963f66afa6 --data ./data --annotations ./gt.json --json
```

**Streamed stdout output:**
```json
{"event": "epoch_progress", "session": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "epoch": 1, "total_epochs": 10, "batch": 14, "total_batches": 150, "loss": 0.3421}
{"event": "epoch_progress", "session": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "epoch": 1, "total_epochs": 10, "batch": 15, "total_batches": 150, "loss": 0.3105}
{"event": "training_completed", "session": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "checkpoints": [".xvisor/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6/checkpoints/checkpoint_epoch_1.pth"]}
```

---

## 📚 Architecture & Theoretical Docs

For in-depth design specifications and trade-off rationales, refer to the documentation:
- [System Design Document](docs/project-design.md): Subsystem breakdowns, data flow diagrams, and MLOps storage hierarchy.
- [Architecture Decisions & Rationale](docs/architecture-decisions.md): Theoretical evaluation of tracking paradigms, mathematical models, and engineering trade-offs.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
