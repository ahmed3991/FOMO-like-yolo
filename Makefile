.PHONY: help install clean train transfer export visualize verify test-imports download-coco download-kaggle

# Python command (use venv if available, otherwise python3)
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo ".venv/bin/python"; else echo "python3"; fi)

# Default target - show help
help:
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "  FOMO YOLOv8 Makefile - Common Tasks"
	@echo "════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install dependencies"
	@echo "  make test-imports     - Verify all imports work"
	@echo ""
	@echo "Dataset Management:"
	@echo "  make download-coco    - Download COCO128 dataset"
	@echo "  make download-kaggle  - Download Kaggle grocery dataset"
	@echo ""
	@echo "Training:"
	@echo "  make train            - Train from scratch (COCO128)"
	@echo "  make train-quick      - Quick test training (2 epochs)"
	@echo "  make transfer         - Transfer learning on new dataset"
	@echo ""
	@echo "Evaluation:"
	@echo "  make visualize        - Visualize predictions"
	@echo "  make verify           - Verify model architecture"
	@echo ""
	@echo "Export & Deployment:"
	@echo "  make export           - Export to all formats (ONNX, TFLite, C)"
	@echo "  make export-onnx      - Export to ONNX only"
	@echo "  make export-tflite    - Export to TFLite with quantization"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            - Clean cache files"
	@echo "  make clean-all        - Clean cache + checkpoints + models"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════"

# Installation
install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Installation complete!"

# Dataset downloads
download-coco:
	@echo "Downloading COCO128 dataset..."
	$(PYTHON) scripts/download_coco128.py --update-config --verify

download-kaggle:
	@echo "Downloading Kaggle grocery dataset..."
	@echo "Note: You need to setup Kaggle credentials first"
	$(PYTHON) scripts/download_kaggle_grocery.py --download --prepare

# Training commands
train:
	@echo "Training model from scratch..."
	$(PYTHON) scripts/train.py \
		--epochs 30 \
		--batch-size 16 \
		--lr 0.001

train-quick:
	@echo "Quick test training (2 epochs)..."
	$(PYTHON) scripts/train.py \
		--epochs 2 \
		--batch-size 8 \
		--quick-test

# Transfer learning
# Usage: make transfer CHECKPOINT=checkpoints/best_model.pth DATA=data/grocery_items CLASSES=43
CHECKPOINT ?= checkpoints/best_model.pth
DATA ?= data/grocery_items
CLASSES ?= 2
EPOCHS ?= 20

transfer:
	@echo "Running transfer learning..."
	@if [ ! -f $(CHECKPOINT) ]; then \
		echo "✗ Error: Checkpoint not found: $(CHECKPOINT)"; \
		echo "  Run 'make train' first or specify: make transfer CHECKPOINT=path/to/checkpoint.pth"; \
		exit 1; \
	fi
	$(PYTHON) scripts/transfer_learning.py \
		--checkpoint $(CHECKPOINT) \
		--train-image-dir $(DATA)/train/images \
		--train-label-dir $(DATA)/train/labels \
		--val-image-dir $(DATA)/valid/images \
		--val-label-dir $(DATA)/valid/labels \
		--num-classes $(CLASSES) \
		--epochs $(EPOCHS) \
		--lr 0.0001 \
		--batch-size 16

# Visualization
# Usage: make visualize CHECKPOINT=checkpoints/best_model.pth IMAGES=data/val/images
IMAGES ?= data/coco128/val/images
NUM_SAMPLES ?= 10

visualize:
	@echo "Generating prediction visualizations..."
	@if [ ! -f $(CHECKPOINT) ]; then \
		echo "✗ Error: Checkpoint not found: $(CHECKPOINT)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/visualize_predictions.py \
		--checkpoint $(CHECKPOINT) \
		--image-dir $(IMAGES) \
		--num-samples $(NUM_SAMPLES) \
		--output-dir predictions \
		--threshold 0.5 \
		--save-individual \
		--create-grid

# Verification
verify:
	@echo "Verifying model architecture..."
	$(PYTHON) -m src.evaluation.verify

test-imports:
	@echo "Testing module imports..."
	@$(PYTHON) -c "from src.models import MicroYOLO; print('✓ Models import OK')"
	@$(PYTHON) -c "from src.datasets import FOMODataset; print('✓ Datasets import OK')"
	@$(PYTHON) -c "from src.training.train import train; print('✓ Training import OK')"
	@$(PYTHON) -c "from src.export import export_to_onnx; print('✓ Export import OK')"
	@$(PYTHON) -c "from src.evaluation import get_centroids_from_heatmap; print('✓ Evaluation import OK')"
	@echo "✓ All imports successful!"

# Export commands
# Usage: make export CHECKPOINT=checkpoints_transfer/best_model_transfer.pth
export:
	@echo "Exporting to all formats..."
	@if [ ! -f $(CHECKPOINT) ]; then \
		echo "✗ Error: Checkpoint not found: $(CHECKPOINT)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/export_model.py \
		--checkpoint $(CHECKPOINT) \
		--all \
		--output-dir models

export-onnx:
	@echo "Exporting to ONNX..."
	$(PYTHON) scripts/export_model.py \
		--checkpoint $(CHECKPOINT) \
		--format onnx \
		--output-dir models

export-tflite:
	@echo "Exporting to TFLite with quantization..."
	$(PYTHON) scripts/export_model.py \
		--checkpoint $(CHECKPOINT) \
		--format tflite \
		--quantize \
		--output-dir models

# Inference example
demo:
	@echo "Running inference demo..."
	$(PYTHON) scripts/inference_example.py

# Cleaning
clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cache cleaned!"

clean-all: clean
	@echo "Cleaning checkpoints and models..."
	rm -rf checkpoints/* checkpoints_transfer/* models/*
	@echo "✓ All cleaned!"

# Development workflow shortcuts
dev-setup: install download-coco test-imports
	@echo "✓ Development environment ready!"

full-workflow: train export visualize
	@echo "✓ Complete training workflow finished!"

# Quick reference
info:
	@echo "Project: FOMO-like YOLOv8 for Microcontrollers"
	@echo "Structure:"
	@echo "  src/datasets/   - Dataset handling"
	@echo "  src/models/     - Model architecture"
	@echo "  src/training/   - Training scripts"
	@echo "  src/export/     - Model export utilities"
	@echo "  src/evaluation/ - Metrics & verification"
	@echo "  scripts/        - CLI utilities"
