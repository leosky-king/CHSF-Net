# CHSF-Net

Research implementation of **CHSF-Net: Cross-Stage Hierarchical Spatial-Frequency Fusion for RGB–Infrared Object Detection**.

CHSF-Net is a dual-stream RGB–infrared multimodal object detection framework built upon YOLO11. It extracts visible-light and infrared features independently, performs hierarchical spatial–frequency feature interaction, and predicts objects through a shared detection head.

> This repository contains research code only. Datasets, trained checkpoints, and experimental outputs are not included.

## Highlights

- Dual-stream RGB–infrared feature extraction
- Cross-stage hierarchical feature fusion
- Spatial and frequency-domain complementary enhancement
- YOLO11-based end-to-end object detection
- Support for paired visible-light and infrared inputs

## Repository Structure

```text
CHSF-Net/
├── improve_multimodal/
│   └── yolo11/
│       └── yolo11-CHSF-Net.yaml    # CHSF-Net architecture
├── ultralytics/
│   └── nn/
│       └── AddModules/
│           └── CHSFNet.py          # CHSF-Net modules
├── data.yaml                       # Dataset configuration
├── train.py                        # Training entry point
└── pyproject.toml                  # Environment dependencies
