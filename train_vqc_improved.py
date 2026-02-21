#!/usr/bin/env python3
"""
Improved VQC Model Training Script
Train the Plant Health Classifier with improved VQC model and save it for later use.
"""

import os
import sys
import traceback
from vqc_improved import ImprovedPlantHealthClassifier

def main():
    """Train and save the improved VQC model"""
    print("Plant Health Classifier - Improved VQC Model Training")
    print("=" * 50)
    
    try:
        # Initialize classifier with 6 qubits for better accuracy
        classifier = ImprovedPlantHealthClassifier(n_qubits=6)
        
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
        
        print("Starting improved VQC model training...")
        print("-" * 40)
        
        # Train the model
        train_acc, test_acc = classifier.train(healthy_folder, diseased_folder)
        
        # Save the trained model
        model_path = "plant_health_model_vqc.pkl"
        classifier.save_model(model_path)
        
        print("-" * 40)
        print(f"Improved VQC model trained successfully!")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Model saved to: {model_path}")
        print("\nNow you can run the Flask app with the improved VQC model.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()