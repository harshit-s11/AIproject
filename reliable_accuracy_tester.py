import numpy as np
import os
from vqc_improved import ImprovedPlantHealthClassifier
from vqc_fast import FastPlantHealthClassifier
from vqc_old import OldPlantHealthClassifier

def run_multiple_training_sessions(model_class, model_name, healthy_folder, diseased_folder, runs=3):
    """
    Run multiple training sessions and collect accuracy statistics
    
    Args:
        model_class: The classifier class to use
        model_name (str): Name of the model for display
        healthy_folder (str): Path to healthy images folder
        diseased_folder (str): Path to diseased images folder
        runs (int): Number of training sessions to run
        
    Returns:
        dict: Statistics about training and test accuracies
    """
    print(f"\n{'='*50}")
    print(f"Running {runs} training sessions for {model_name} model")
    print(f"{'='*50}")
    
    train_accuracies = []
    test_accuracies = []
    
    for i in range(runs):
        print(f"\n--- Session {i+1}/{runs} ---")
        try:
            # Create model instance
            if model_name == "Fast":
                model = model_class(n_qubits=4)
            elif model_name == "Old":
                model = model_class(n_qubits=2)
            else:  # Improved
                model = model_class(n_qubits=6)
            
            # Train model
            train_acc, test_acc = model.train(healthy_folder, diseased_folder)
            
            # Store accuracies
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"Session {i+1} - Accuracy: {train_acc:.4f}")
            
        except Exception as e:
            print(f"Session {i+1} failed with error: {str(e)}")
            continue
    
    # Calculate statistics
    if len(train_accuracies) > 0:
        train_mean = np.mean(train_accuracies)
        train_std = np.std(train_accuracies)
        test_mean = np.mean(test_accuracies)
        test_std = np.std(test_accuracies)
        
        print(f"\n--- {model_name} Model Accuracy Statistics ---")
        print(f"Accuracy: {train_mean:.4f} ± {train_std:.4f}")
        print(f"Number of successful sessions: {len(train_accuracies)}/{runs}")
        
        return {
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'successful_runs': len(train_accuracies)
        }
    else:
        print(f"\n--- {model_name} Model: All sessions failed ---")
        return {
            'train_accuracies': [],
            'test_accuracies': [],
            'train_mean': 0.0,
            'train_std': 0.0,
            'test_mean': 0.0,
            'test_std': 0.0,
            'successful_runs': 0
        }

def compare_models(healthy_folder="Dataset/Tomato_Healthy", diseased_folder="Dataset/Tomato_diseased", runs=3):
    """
    Compare Fast and Basic models with multiple training sessions
    
    Args:
        healthy_folder (str): Path to healthy images folder
        diseased_folder (str): Path to diseased images folder
        runs (int): Number of training sessions per model
    """
    print("Quantum Model Accuracy Reliability Tester")
    print("This script runs multiple training sessions to get reliable accuracy measurements")
    
    # Check if dataset folders exist
    if not os.path.exists(healthy_folder) or not os.path.exists(diseased_folder):
        print(f"Error: Dataset folders not found!")
        print(f"Expected folders: {healthy_folder} and {diseased_folder}")
        return
    
    # Test Fast model
    print("\n" + "="*60)
    print("TESTING FAST MODEL")
    print("="*60)
    fast_stats = run_multiple_training_sessions(
        FastPlantHealthClassifier, 
        "Fast", 
        healthy_folder, 
        diseased_folder, 
        runs
    )
    
    # Test Basic model
    print("\n" + "="*60)
    print("TESTING BASIC MODEL")
    print("="*60)
    basic_stats = run_multiple_training_sessions(
        OldPlantHealthClassifier, 
        "Old", 
        healthy_folder, 
        diseased_folder, 
        runs
    )
    
    # Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Fast Model   - Accuracy: {fast_stats['train_mean']:.4f} ± {fast_stats['train_std']:.4f}")
    print(f"Old Model  - Accuracy: {basic_stats['train_mean']:.4f} ± {basic_stats['train_std']:.4f}")
    
    if fast_stats['train_mean'] > basic_stats['train_mean']:
        print("Fast model has higher average accuracy")
    elif basic_stats['train_mean'] > fast_stats['train_mean']:
        print("Basic model has higher average accuracy")
    else:
        print("Both models have similar average accuracy")

if __name__ == "__main__":
    # Run comparison with 3 sessions per model
    compare_models(runs=3)