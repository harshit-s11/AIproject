# Improved VQC Version

This folder contains both the improved and fast Variational Quantum Classifier (VQC) implementations.

## Files

- `vqc_improved.py` - Improved VQC implementation with enhanced quantum circuits (higher accuracy, longer training)
- `vqc_fast.py` - Fast VQC implementation with reduced complexity (quicker training, good accuracy)
- `train_vqc_improved.py` - Training script for the improved VQC model
- `train_vqc_fast.py` - Training script for the fast VQC model
- `app_vqc.py` - Flask web application that automatically uses the best available model
- `requirements.txt` - Python dependencies for this version

## Usage

### Quick Start (Recommended)
1. Train the fast VQC model (much quicker):
   ```bash
   python train_vqc_fast.py
   ```

### Full Training (Higher Accuracy)
1. Train the improved VQC model:
   ```bash
   python train_vqc_improved.py
   ```

2. Run the web application:
   ```bash
   python app_vqc.py
   ```

3. Access the web interface at http://localhost:5001

## Improvements

### Fast VQC Model (`vqc_fast.py`)
- Reduced quantum circuit complexity (4 qubits vs 6)
- Simpler feature map and ansatz configurations
- Lower resolution image processing (32x32 vs 64x64)
- Limited training dataset for faster execution
- Training time: 5-15 minutes

### Improved VQC Model (`vqc_improved.py`)
- Enhanced quantum circuit design with more repetitions
- Better feature map and ansatz configurations
- Improved training process with callbacks
- Higher resolution image processing (64x64)
- Training time: 30+ minutes