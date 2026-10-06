# Project Design: Human Video Tracking System

## 1. System Overview

This system is an engineering framework for training, evaluating, maintaining, and exporting computer vision models for **Multi-Object Tracking (MOT) of humans in video**.

The core focus of this system is **lifecycle management of tracking models**:
- Training object detection models from scratch or from existing checkpoints.
- Persisting complete training states to allow seamless session resumption.
- Decoupled testing pipelines to evaluate Detection ($mAP$) and Tracking ($MOTA$, $IDF1$) independently.
- Exporting trained models to portable formats (e.g., ONNX) for deployment in external applications across different programming languages and runtimes.
- Providing user interfaces that support both scriptable automation and interactive guided execution.

---

## 2. Abstract System Architecture

The architecture is organized into logical subsystems with decoupled responsibilities:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                            │
│           [Scriptable CLI Subcommands]   [Interactive TUI]            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Execution Engine                                │
│  ┌────────────────────┐ ┌────────────────────┐ ┌─────────────────────┐ │
│  │   Training Engine  │ │  Detection Tester  │ │   Tracking Tester   │ │
│  └─────────┬──────────┘ └─────────┬──────────┘ └──────────┬──────────┘ │
│            │                      │                       │            │
│  ┌─────────┴──────────┐           │            ┌──────────┴──────────┐ │
│  │ Checkpoint Manager │           │            │    Model Exporter   │ │
│  └────────────────────┘           │            └─────────────────────┘ │
└───────────────────┬───────────────┴───────────────────────┬────────────┘
                    │                                       │
                    ▼                                       ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│             Data Layer                │ │         Model Layer          │
│  - Dataset Ingestion Adapters         │ │  - Object Detector           │
│    (COCO JSON, MOT TXT)               │ │  - Appearance ReID Embedder  │
│  - Data Augmentation & Preprocessing  │ │  - Tracking Associator       │
│  - Batch Loading & Collation          │ │    (Motion + Visual Matching)│
└───────────────────────────────────────┘ └──────────────────────────────┘
```

---

## 3. Subsystem Specifications

### 3.1 User Interface Subsystem
Provides two entry modes:
1. **Scriptable Command-Line Interface (CLI)**: Enables non-interactive execution with structured subcommands (`train`, `resume`, `test-detection`, `test-tracking`, `export`). Suitable for automated scripts and headless environments.
2. **Interactive Terminal User Interface (TUI)**: A menu-driven flow that interactively prompts for actions, dataset paths, checkpoint files, and hyperparameters with validation.

### 3.2 Data Ingestion & Transformation Subsystem
Abstracts data loading to support multiple annotation schemes without changing engine logic:
- **Annotation Adapters**:
  - *COCO Format Adapter*: Parses standard JSON annotations containing image metadata, bounding box coordinates, and category labels.
  - *MOT Format Adapter*: Parses text-based sequence annotations containing frame indices, object IDs, and spatial boxes.
- **Preprocessing Pipeline**: Resizing, normalization, and training data augmentations (e.g., flips, photometric adjustments).
- **Batch Generator**: Combines images and variable-length bounding box targets into tensor batches.

### 3.3 Model & Tracking Subsystem
Implements the multi-stage tracking pipeline:
- **Human Detector**: Two-stage deep neural network that locates human bounding boxes and confidence scores within single video frames.
- **Appearance Feature Extractor (ReID)**: A deep convolutional network that maps cropped person image patches to normalized 1D embedding vectors.
- **State Estimator (Kalman Filter)**: Maintains position, scale, and velocity vectors for each tracked target, projecting expected locations into subsequent frames.
- **Data Associator**: Computes cost matrices combining spatial/motion distance and visual cosine similarity. Solves the bipartite matching problem to match existing tracks with new detections, managing track birth, confirmation, and deletion.

### 3.4 Execution Engine Subsystem
Orchestrates training, evaluation, and persistence:
- **Training Engine**: Drives forward passes, computes multitask losses (classification and bounding box regression), performs backpropagation, and steps optimizers and schedulers. Automatically binds to available GPU compute or falls back to CPU.
- **Checkpoint Manager**: Serializes and deserializes model weights, optimizer momentum buffers, learning rate scheduler state, epoch counters, and validation metrics. Manages both full-state checkpoints (for resumption) and lightweight models (for inference).
- **Detection Evaluator**: Evaluates frame-level detector performance independently from tracking, computing Mean Average Precision ($mAP_{50}$, $mAP_{50:95}$).
- **Tracking Evaluator**: Evaluates end-to-end video tracking performance, computing Multi-Object Tracking Accuracy ($MOTA$), Identification F1 score ($IDF1$), and ID switch counts ($IDSW$).
- **Model Exporter**: Translates model computation graphs and weights into standardized, runtime-agnostic formats (e.g., ONNX) for deployment outside the primary development environment.

---

## 4. Core Workflows

### 4.1 Training & State Resumption Workflow
```
[Start Session]
      │
      ├── Mode: Fresh Training
      │     └── Initialize weights from pretrained base
      │     └── Initialize new optimizer & learning rate scheduler
      │
      └── Mode: Resume Training
            └── Load checkpoint
            └── Restore model weights
            └── Restore optimizer momentum buffers
            └── Restore scheduler state and start epoch
      │
      ▼
[Training Loop (Epoch by Epoch)]
      │
      ├── Forward pass on training batches
      ├── Aggregate detection losses
      ├── Backward pass & gradient updates
      └── Evaluate validation loss
      │
      ▼
[Checkpoint Serialization]
      ├── Full State Checkpoint (Weights + Optimizer + Scheduler + Epoch)
      └── Best Performance Checkpoint (Saved on metric improvements)
```

### 4.2 Decoupled Evaluation Workflow
```
                   [Trained Checkpoint]
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
  [Detection Evaluation]          [Tracking Evaluation]
  - Requires: Images + Boxes      - Requires: Ordered video frames + Track IDs
  - Ignores temporal consistency  - Evaluates temporal track identity
  - Output: mAP50, mAP50:95       - Output: MOTA, IDF1, ID Switches
```

### 4.3 Export & External Deployment Workflow
```
[PyTorch Checkpoint]
        │
        ▼
[Model Exporter] ──► Trace computation graph with dummy input
        │
        ▼
[Portable Format (.onnx)]
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
 [C++ / Native App]      [C# / .NET App]       [Web / Browser]
 (via ONNX Runtime)     (via ONNX Runtime)   (via ONNX Runtime Web)
```
