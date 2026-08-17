#!/usr/bin/env python3
"""
Quantum Plant Health Classifier - Model Training Script

Trains a Variational Quantum Classifier (VQC) using Qiskit and Scikit-learn
on plant leaf images to classify healthy vs diseased leaves.
"""

import os
import sys
import traceback
from vqc_model import PlantHealthClassifier


def main():
    print("=" * 60)
    print("  Quantum Plant Health Classifier - Model Training")
    print("=" * 60)

    healthy_folder = os.path.join("Dataset", "Tomato_Healthy")
    diseased_folder = os.path.join("Dataset", "Tomato_diseased")
    model_output_path = "plant_health_model.pkl"

    if not os.path.exists(healthy_folder):
        print(f"Error: Healthy dataset folder '{healthy_folder}' not found.")
        sys.exit(1)

    if not os.path.exists(diseased_folder):
        print(f"Error: Diseased dataset folder '{diseased_folder}' not found.")
        sys.exit(1)

    try:
        print("\n1. Initializing 4-Qubit PlantHealthClassifier...")
        classifier = PlantHealthClassifier(n_qubits=4)

        print("\n2. Training VQC Model with Stratified 80/20 Train/Test Split...")
        train_acc, test_acc = classifier.train(
            healthy_folder=healthy_folder,
            diseased_folder=diseased_folder,
            test_size=0.2,
            random_state=42,
            max_samples=50
        )

        print(f"   - Verified Training Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"   - Verified Test Accuracy:     {test_acc:.4f} ({test_acc * 100:.2f}%)")

        print(f"\n3. Saving trained model to '{model_output_path}'...")
        classifier.save_model(model_output_path)
        print(f"   Model successfully saved to {model_output_path}")

        print("\n" + "=" * 60)
        print("Training complete. You can now start the web app with: python app.py")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
