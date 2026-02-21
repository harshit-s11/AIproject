#!/usr/bin/env python3
"""
Basic VQC Model Training Script
Train a Plant Health Classifier with reduced VQC model performance.
"""

import os
import sys
import traceback
from vqc_old import OldPlantHealthClassifier

def main():
    """Train and save the basic VQC model"""
    print("Plant Health Classifier - Basic VQC Model Training")
    print("=" * 50)
    print("NOTE: This model uses reduced complexity for faster training with basic accuracy")
    print("-" * 50)
    
    try:
        # Initialize classifier with minimal qubits for basic performance
        classifier = OldPlantHealthClassifier(n_qubits=2)
        
        # Define dataset paths
        healthy_folder = "Dataset/Tomato_Healthy"
        diseased_folder = "Dataset/Tomato_diseased"
        
        # Check if dataset folders exist
        if not os.path.exists(healthy_folder):
            print(f"Error: Healthy folder '{healthy_folder}' not found!")
            return
        
        if not os.path.exists(diseased_folder):
            print(f"Error: Diseased folder '{diseased_folder}' not found!")
            return
        
        print("Starting basic VQC model training...")
        print("-" * 40)
        
        # Train the model
        train_acc, test_acc = classifier.train(healthy_folder, diseased_folder)
        
        # Save the trained model
        model_path = "plant_health_model_old.pkl"
        classifier.save_model(model_path)
        
        print("-" * 40)
        print(f"Basic VQC model trained successfully!")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Model saved to: {model_path}")
        print("\nThis model uses reduced complexity for basic performance.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()