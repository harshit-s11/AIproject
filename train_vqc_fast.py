#!/usr/bin/env python3
"""
Fast VQC Model Training Script
Train a faster Plant Health Classifier with reduced VQC model and save it for later use.
"""

import os
import sys
import traceback
from vqc_fast import FastPlantHealthClassifier

def main():
    """Train and save the fast VQC model"""
    print("Plant Health Classifier - Fast VQC Model Training")
    print("=" * 50)
    
    try:
        # Initialize classifier with 4 qubits for faster training
        classifier = FastPlantHealthClassifier(n_qubits=4)
        
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
        
        print("Starting fast VQC model training...")
        print("-" * 40)
        
        # Train the model
        train_acc, test_acc = classifier.train(healthy_folder, diseased_folder)
        
        # Save the trained model
        model_path = "plant_health_model_fast.pkl"
        classifier.save_model(model_path)
        
        print("-" * 40)
        print(f"Fast VQC model trained successfully!")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Model saved to: {model_path}")
        print("\nNow you can run the Flask app with the fast VQC model.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()