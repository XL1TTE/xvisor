# Architecture Decisions & Theoretical Rationale

This document records the architectural choices evaluated during system design. It details the theoretical foundations of all candidate approaches, their respective trade-offs, and the rationale behind the selected design.

---

## 1. Tracking Paradigm

### Context
Multi-Object Tracking (MOT) in video requires identifying objects of interest in each frame and maintaining persistent identity labels across time.

### Options Considered

#### Option A: Tracking-by-Detection (Selected)
- **Theoretical Mechanism**: The task is split into two independent, sequential sub-problems:
  1. *Detection*: An independent detector processes each frame $t$ as an isolated still image to output candidate bounding boxes $B_t = \{b_1, b_2, \dots, b_n\}$.
  2. *Data Association*: An independent tracking algorithm matches detections from frame $t$ to existing track trajectories from frame $t-1$ based on spatial proximity, motion estimation, and visual similarity.
- **Advantages**:
  - *Modularity*: Detectors and trackers can be developed, trained, swapped, and benchmarked independently.
  - *Data flexibility*: The detector can be trained on standard static image datasets (e.g., COCO, CrowdHuman) without requiring expensive video-level tracking labels.
  - *Maturity*: Backed by well-established algorithms (SORT, DeepSORT, ByteTrack) with predictable behavior.
- **Disadvantages**:
  - *Cascading failure*: The tracker relies entirely on detector output. A missed detection cannot be recovered by the tracker, and false positive detections generate spurious tracks.

#### Option B: Joint Detection and Tracking (One-Shot / End-to-End)
- **Theoretical Mechanism**: A single neural network simultaneously outputs bounding box coordinates, class labels, and either identity embedding vectors or temporal motion offsets in a single unified forward pass (e.g., FairMOT, JDE).
- **Advantages**:
  - *Computational efficiency*: Feature extraction is shared between the detection head and the identity/tracking head, avoiding redundant passes over the image.
  - *Inference speed*: Typically runs faster than multi-stage systems running independent ReID networks for every detected crop.
- **Disadvantages**:
  - *Optimization conflict*: Balancing detection loss (which requires features invariant to intra-class differences) and identity ReID loss (which requires features sensitive to subtle individual differences) creates competition in shared network backbones.
  - *Dataset constraints*: Requires video datasets with simultaneous bounding box and dense identity annotations.

#### Option C: Transformer-Based MOT (Direct Temporal Query Tracking)
- **Theoretical Mechanism**: Uses transformer attention mechanisms (e.g., TrackFormer, MOTR) where object queries persist across video frames and attend to both spatial feature maps and temporal history.
- **Advantages**:
  - Eliminates hand-crafted association rules, Kalman filters, and explicit bipartite matching heuristics.
  - Natural handling of long-term occlusions via attention over history tokens.
- **Disadvantages**:
  - Significant computational and memory footprints, requiring high-end GPUs.
  - Long training schedules and high data hunger.
  - Difficult to inspect or debug when associations fail.

### Decision & Rationale
**Selected: Option A (Tracking-by-Detection)**.  
It provides clear architectural boundaries, allows the detector to be trained independently on image data, simplifies debugging and testing, and avoids the heavy hardware constraints and multi-task loss balancing issues inherent to joint or transformer-based models.

---

## 2. Data Association Method

### Context
Within the Tracking-by-Detection paradigm, an algorithm must assign new detections in frame $t$ to active tracks from frame $t-1$.

### Options Considered

#### Option 1: Motion and Spatial Only (SORT / ByteTrack style)
- **Theoretical Mechanism**: A Kalman Filter models the continuous velocity and position of each target. At frame $t$, the filter predicts the expected bounding box. dAssociation cost is calculated purely using spatial overlap (Intersection over Union, IoU) or generalized IoU (GIoU). Optimal assignment is solved via the Hungarian algorithm.
- **Advantages**:
  - Zero neural network overhead for the tracking step; executes in fractions of a millisecond on CPU.
  - No training required for the tracker component.
- **Disadvantages**:
  - Susceptible to identity switches when two humans cross paths or occlude one another.
  - Incapable of re-identifying targets that leave the frame or remain completely occluded for multiple seconds (motion-only models lose track continuity once the prediction drifts).

#### Option 2: Motion + Visual Appearance (DeepSORT / BoT-SORT style) (Selected)
- **Theoretical Mechanism**: Combines spatial motion estimation (Kalman Filter) with visual feature representations (Re-Identification / ReID network). For every detection, cropped image pixels are passed through a feature extractor to obtain a normalized 1D embedding vector $f \in \mathbb{R}^d$. The assignment cost between track $i$ and detection $j$ combines:
  1. *Mahalanobis distance* (motion plausibility based on Kalman filter uncertainty).
  2. *Cosine distance* between visual feature embeddings ($1 - \frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}$).
- **Advantages**:
  - Robust identity preservation during crossing trajectories.
  - Enables long-term re-identification: a person who exits and re-enters the scene can be recognized by appearance.
- **Disadvantages**:
  - Higher computational cost: running a convolutional network on every detected human crop introduces latency.

### Decision & Rationale
**Selected: Option 2 (Motion + Visual Appearance / DeepSORT style)**.  
Visual embeddings provide necessary resilience against occlusions and trajectory crossings in crowded human scenes, yielding more reliable tracking behavior than motion-alone heuristics.

---

## 3. Training Scope & Model Division

### Context
With a two-component model (Detector + ReID Embedder), the scope of what gets trained must be defined.

### Options Considered

#### Approach 1: Train Both Detector and ReID Network from Scratch
- **Theoretical Mechanism**: Prepare two distinct datasets: a bounding box detection dataset and an identity-labeled person re-identification dataset (triplet loss / contrastive loss). Train both models independently.
- **Trade-off**: High training workload and complex dataset preparation requirements.

#### Approach 2: Train Detector, Use Pretrained ReID Network (Selected)
- **Theoretical Mechanism**: Train or fine-tune the human detector on target domain imagery. Use an established, publicly available model pretrained on large-scale person re-identification benchmarks (e.g., Market-1501, DukeMTMC) for visual feature extraction.
- **Trade-off**: ReID features are generic rather than fine-tuned to specific camera lighting, but the training pipeline remains focused, stable, and manageable.

### Decision & Rationale
**Selected: Approach 2 (Train Detector, Pretrained ReID)**.  
This allows full engineering focus on building a robust training loop, checkpointing engine, and evaluation suite for the detector without diluting effort across two independent training systems simultaneously.

---

## 4. Framework & Level of Abstraction

### Options Considered

#### Option A: High-Level Frameworks (e.g., Ultralytics YOLO)
- **Characteristics**: Wraps training, validation, export, and inference behind single CLI commands or high-level function calls.
- **Trade-off**: Fast prototyping, but obscures tensor operations, internal loss formulations, optimizer buffer mechanics, and custom checkpoint structures.

#### Option B: Native PyTorch (Selected)
- **Characteristics**: Direct implementation of the forward pass, loss extraction, gradient backward pass, optimizer step, learning rate scheduler updates, and manual dictionary serialization.
- **Trade-off**: Requires more boilerplate code, but grants complete control over data loading, custom metrics, checkpoint serialization, and runtime device management.

### Decision & Rationale
**Selected: Option B (Native PyTorch)**.  
The goal is deep understanding and full control over training loops, state resumption, and model lifecycle mechanics.

---

## 5. Detector Architecture

### Options Considered

#### Option 1: Faster R-CNN (Selected)
- **Theoretical Mechanism**: A two-stage detector.
  - Stage 1: Region Proposal Network (RPN) slides over convolutional feature maps to propose candidate object regions.
  - Stage 2: Fast R-CNN head applies RoI Pooling / RoI Align to extract features from proposals and computes class probabilities alongside bounding box offsets.
- **Advantages**:
  - Available out-of-the-box in `torchvision` with pretrained backbones (ResNet-50 FPN, MobileNetV3).
  - Clean native PyTorch interface: automatically computes the multi-task loss dictionary during training mode and outputs decoded box predictions during evaluation mode.
  - Highly stable convergence during training.
- **Disadvantages**: Slower inference speed compared to modern single-stage detectors.

#### Option 2: Single-Stage Detectors (SSD / RetinaNet / YOLO)
- **Theoretical Mechanism**: Directly predicts bounding boxes and class confidences across dense anchor grids in a single forward pass without a separate proposal stage.
- **Disadvantages**: Implementing custom anchor assignment, focal loss handling, or non-maximum suppression from scratch in native PyTorch introduces substantial complexity unrelated to the primary tracking objectives.

### Decision & Rationale
**Selected: Option 1 (Faster R-CNN)**.  
It provides standard, production-grade object detection in native PyTorch with clean loss calculation, allowing the project to focus on training orchestration, checkpoint management, and tracking integration.

---

## 6. Dataset Format Support

### Options Considered

#### Format 1: COCO JSON
- Single structured JSON containing `images` lists, `categories`, and `annotations` ($[x, y, width, height]$).
- Standard for object detection benchmarks.

#### Format 2: MOTChallenge TXT
- Sequential text files: `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>`.
- Standard for multi-object tracking sequences.

### Decision & Rationale
**Selected: Support both formats via an adapter interface**.  
Supporting COCO JSON enables training on standard object detection datasets. Supporting MOTChallenge TXT enables evaluation on actual video tracking benchmarks. An abstract dataset interface decouples the engine from the underlying file structure.

---

## 7. Checkpointing & State Persistence Strategy

### Options Considered

#### Strategy 1: Naive Weights-Only Saving (`model.state_dict()`)
- Only serializes neural network parameter tensors.
- **Limitation**: When training is resumed, the optimizer starts with empty momentum and variance buffers. In optimizers like Adam or SGD with momentum, resetting these buffers causes abrupt gradient shifts and learning rate discontinuities that degrade model performance.

#### Strategy 2: Complete Training State Persistence (Selected)
- Serializes a dictionary containing:
  - `model_state_dict`: Neural network weights.
  - `optimizer_state_dict`: First and second moment buffers.
  - `scheduler_state_dict`: Current learning rate decay step.
  - `epoch`: Last completed epoch index.
  - `best_metric`: Validation performance record.
- **Advantages**: Training can be halted and resumed identically to an uninterrupted run. Also provides an export utility to strip non-weight metadata for inference deployment.

### Decision & Rationale
**Selected: Strategy 2 (Complete Training State Persistence)**.  
Essential for reliable, production-grade machine learning workflows.

---

## 8. Decoupled Testing Strategy

### Context
Evaluating a tracking system requires measuring performance, but combining detection and tracking into a single evaluation creates ambiguity when diagnosing failures.

### Decision & Rationale
**Selected: Two independent evaluation pipelines**:
1. **Detection Testing ($mAP$)**: Evaluates the detector on static test images. Measures Mean Average Precision at IoU thresholds ($mAP_{50}$, $mAP_{50:95}$). Identifies if the detector is missing humans or generating false positives.
2. **Tracking Testing ($MOTA$, $IDF1$, $IDSW$)**: Evaluates the full pipeline on video sequences with ground-truth track IDs. Measures tracking consistency, trajectory fragmentation, and identity swaps over time.

Decoupling these tests ensures clear attribution of errors: detection failures (poor box localization) vs. association failures (erratic track matching).

---

## 9. Hardware Portability Strategy

### Decision & Rationale
**Selected: Automatic hardware abstraction (CUDA with graceful CPU fallback)**.  
The system dynamically queries `torch.cuda.is_available()`. If a CUDA GPU is present, models and tensors are assigned to `cuda:0` with appropriate memory settings. If absent, the system transparently defaults to `cpu`. This guarantees identical functionality across dedicated GPU workstations and standard CPU environments.

---

## 10. Model Export & Interoperability

### Options Considered

#### Approach 1: Python-Only Execution (`.pth` / `.pt`)
- Requires the Python runtime, PyTorch installation, and original model definitions.
- Not practical for embedding into external C++, C#, mobile, or web applications.

#### Approach 2: Portable Format Export (ONNX / TorchScript) (Selected)
- Translates the PyTorch model graph into **ONNX (Open Neural Network Exchange)**.
- **Advantages**:
  - ONNX models run in C++, C#, Go, Java, and JavaScript via **ONNX Runtime** without a Python interpreter.
  - Allows hardware acceleration using vendor engines like NVIDIA TensorRT or Intel OpenVINO.

### Decision & Rationale
**Selected: Approach 2 (Export to ONNX)**.  
Provides a clean bridge between model training in Python and deployment in external applications.
