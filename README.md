# 😷 Face Mask Detection in Public Places (YOLOv8-FD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-FD-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

An improved real-time face mask detection and compliance monitoring system based on **YOLOv8-FD** (Face mask Detection), designed for public health management. It accurately distinguishes between **correct mask wearing**, **incorrect mask wearing (e.g., under nose/chin)**, and **no mask**, even under dense crowds, extreme lighting, and partial occlusions.

---

## ✨ Key Features

- **🎯 Granular 3-State Classification**: Distinguishes between `With Mask`, `Without Mask`, and `Mask Worn Incorrectly`.
- **⚙️ YOLOv8-FD Architecture Improvements**:
  - **Focal Loss Integration**: Solves extreme class imbalance for the rare "incorrect mask" class.
  - **CBAM Attention Mechanism**: Added Convolutional Block Attention Modules (Channel & Spatial Attention) to the PAN-FPN neck for superior feature extraction under occlusions.
  - **Uncertainty Logic Inference Layer**: Uses confidence threshold heuristics ($\tau_{high}=0.70, \tau_{low}=0.40$) for fine-grained compliance state decisions.
- **⚡ Dynamic Inference Optimization**: Adapts runtime model resolution and complexity based on real-time scene analysis, boosting average throughput up to **68 FPS**.
- **📹 Temporal Tracking & Behavior Analysis**: Integrated with **ByteTrack** for consistent object ID persistence and continuous compliance behavior tracking across video frames.
- **🖥️ User-Friendly PyQt5 GUI**: Full-featured graphical interface supporting static image auditing, pre-recorded video stream analysis, and low-latency live camera feeds.
- **🔒 Privacy-Aware Design**: Optional real-time facial anonymization (blurring non-compliance features) and local edge processing for strict data sovereignty.

---

## 📊 Performance Comparison

Evaluated on the **Kaggle Face Mask Detection Dataset** (8,531 images) using an NVIDIA RTX 4090 GPU:

| Model Architecture | mAP@0.5 | Precision | Recall | F1 Score | Speed (FPS) | Parameters (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Faster R-CNN (ResNet-50) | 0.843 | 0.821 | 0.802 | 0.811 | 12 | 41.5 |
| YOLOv8m (Baseline) | 0.891 | 0.868 | 0.854 | 0.861 | 52 | 25.9 |
| **YOLOv8-FD (Ours)** | **0.912** | **0.882** | **0.893** | **0.887** | **45 (Dynamic: 68)** | **26.3** |

### Per-Class Average Precision ($AP_{0.5}$)
- **With Mask**: `0.945`
- **Without Mask**: `0.916`
- **Incorrectly Worn**: `0.815` *(+7.4% improvement over baseline YOLOv8m)*

---

## 🏗️ System Architecture

```
                       ┌────────────────────────┐
                       │   Input Image/Stream   │
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │   CSPDarknet Backbone  │
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │ PAN-FPN Neck + CBAM    │
                       └───────────┬────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             ▼                     ▼                     ▼
     Decoupled Head         Decoupled Head        Decoupled Head
     (Small Objects)        (Medium Objects)      (Large Objects)
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │   Focal Loss Engine    │
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │ ByteTrack + GUI Logic  │
                       └────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8 / PyTorch 2.0+ (GPU acceleration strongly recommended)

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/your-username/Face-Mask-Detection-YOLOv8.git
cd Face-Mask-Detection-YOLOv8

# Create conda environment
conda create -n mask-det python=3.8 -y
conda activate mask-det

# Install dependencies
pip install -r requirements.txt
```

### 2. Pretrained Weights
Download the pre-trained `YOLOv8-FD` weights and place them in the `weights/` directory:
```bash
mkdir weights
# Place your yolov8_fd.pt inside the weights/ folder
```

### 3. Running the GUI Application

Launch the desktop interface built with PyQt5:

```bash
python main.py
```

---

## 💡 Usage Guide

### Graphical User Interface (GUI)
1. **Static Image Mode**: Load single images or batch folders to perform compliance checks and export logs.
2. **Video Stream Mode**: Analyze pre-recorded video files (`.mp4`, `.avi`) with persistent ID tracking provided by ByteTrack.
3. **Webcam Feed**: Real-time camera monitoring with direct FPS and live violation stats display.

### Command Line Interface (CLI)
You can also run inference via Python script:

```python
from models.detector import MaskDetector

# Initialize Detector
detector = MaskDetector(weights="weights/yolov8_fd.pt", conf_thresh=0.40)

# Run prediction on image
detections, annotated_img = detector.predict("test.jpg", dynamic_mode=True)
```

---

## 🛠️ Training & Custom Dataset

To retrain or fine-tune YOLOv8-FD on your custom dataset:

1. Prepare dataset in YOLO format with `yaml` dataset structure:
   ```yaml
   path: ./dataset
   train: images/train
   val: images/val
   test: images/test

   names:
     0: with_mask
     1: without_mask
     2: mask_worn_incorrect
   ```

2. Start training:
   ```bash
   python train.py --config configs/yolov8_fd.yaml --data dataset.yaml --epochs 150 --batch 32 --img 640
   ```

---

## 📜 Citation

If you find this repository or dissertation research useful for your work, please cite:

```bibtex
@article{chen2026facemask,
  title={Face Mask Detection in Public Places Using Improved YOLOv8-FD},
  author={Yang, Chen},
  journal={South China Normal University Thesis},
  year={2026}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).
