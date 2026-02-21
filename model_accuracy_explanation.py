#!/usr/bin/env python3
"""
Model Accuracy Calculation Explanation
This script explains exactly how accuracy is calculated in the Fast and Basic VQC models.
"""

def explain_accuracy_calculation():
    """
    Explain how accuracy is calculated in the VQC models
    """
    print("VQC MODEL ACCURACY CALCULATION EXPLANATION")
    print("=" * 50)
    
    print("\nHOW ACCURACY IS CALCULATED IN YOUR MODELS")
    print("-" * 40)
    
    explanation = """
In both vqc_fast.py and vqc_old.py, accuracy is calculated in the train() method:

1. AFTER TRAINING THE MODEL:
   # Get predictions on training data
   y_pred_train = self.model.predict(X_train)
   
   # Get predictions on test data
   y_pred_test = self.model.predict(X_test)

2. CALCULATE ACCURACY:
   from sklearn.metrics import accuracy_score
   
   # Training accuracy
   train_accuracy = accuracy_score(y_train, y_pred_train)
   
   # Test accuracy  
   test_accuracy = accuracy_score(y_test, y_pred_test)

3. THE ACCURACY_SCORE FUNCTION:
   - Formula: (Number of Correct Predictions) / (Total Number of Predictions)
   - Example: If model correctly predicts 17 out of 20 test samples
   - Accuracy = 17/20 = 0.85 or 85%

4. WHERE THIS HAPPENS IN YOUR CODE:
   
   In the train() method of both models:
   -------------------------------------
   # Evaluate model
   y_pred_train = self.model.predict(X_train)     # Get model predictions
   y_pred_test = self.model.predict(X_test)       # Get model predictions
   
   train_accuracy = accuracy_score(y_train, y_pred_train)  # Calculate training accuracy
   test_accuracy = accuracy_score(y_test, y_pred_test)     # Calculate test accuracy
   
   print(f"Training Accuracy: {train_accuracy:.4f}")        # Display results
   print(f"Test Accuracy: {test_accuracy:.4f}")             # Display results

5. WHAT THE NUMBERS MEAN:
   - Training Accuracy: How well the model performs on data it learned from
   - Test Accuracy: How well the model performs on new, unseen data (more important)
   - Higher values are better (1.0 = 100% accuracy, 0.0 = 0% accuracy)

6. DIFFERENCES BETWEEN MODELS:
   - Fast Model: Uses more qubits and complex circuits, generally higher accuracy
   - Basic Model: Uses fewer qubits and simpler circuits, generally lower accuracy
   """
    
    print(explanation)
    
    print("\nEXAMPLE CALCULATION:")
    print("-" * 20)
    print("If your test data has 20 images:")
    print("  - True labels: [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1]")
    print("  - Model predictions: [0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,1,1,1,0,1]")
    print("  - Correct predictions: 15 (75%)")
    print("  - Accuracy = 15/20 = 0.75 or 75%")
    
    print("\nWHEN YOU TRAIN YOUR MODELS:")
    print("-" * 30)
    print("Run: python train_vqc_fast.py")
    print("Output will show: 'Test Accuracy: 0.XXXX'")
    print("Run: python train_vqc_old.py") 
    print("Output will show: 'Test Accuracy: 0.XXXX'")
    
    print("\nACCESSING ACCURACY AFTER TRAINING:")
    print("-" * 35)
    print("The train() method returns both accuracies:")
    print("  train_acc, test_acc = classifier.train(healthy_folder, diseased_folder)")
    print("  print(f'Test Accuracy: {test_acc:.4f}')")

if __name__ == "__main__":
    explain_accuracy_calculation()