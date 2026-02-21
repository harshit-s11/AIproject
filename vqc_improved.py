import numpy as np
from PIL import Image
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

# Try to import scikit-learn components
try:
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    train_test_split = None
    accuracy_score = None
    SKLEARN_AVAILABLE = False

# Try to import Qiskit components
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms import VQC
    QISKIT_AVAILABLE = True
except ImportError:
    ZZFeatureMap = None
    RealAmplitudes = None
    VQC = None
    QISKIT_AVAILABLE = False

class ImprovedPlantHealthClassifier:
    def __init__(self, n_qubits=6):
        """
        Initialize the Plant Health Classifier with improved VQC
        
        Args:
            n_qubits (int): Number of qubits to use for the quantum circuit
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for this classifier")
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for this classifier")
            
        self.n_qubits = n_qubits
        self.pca = None
        self.model = None
        self.is_trained = False
        # Store training data for quick retraining
        self.X_train = None
        self.y_train = None
        
    def preprocess_image(self, image_path, img_size=(64, 64)):
        """
        Preprocess a single image for classification
        
        Args:
            image_path (str): Path to the image file
            img_size (tuple): Size to resize the image to
            
        Returns:
            np.array: Preprocessed image array
        """
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize(img_size)  # Resize to larger size for more features
        
        # Convert to numpy array and flatten
        img_array = np.array(img)
        img_flat = img_array.flatten()
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_flat.astype('float32') / 255.0
        
        return img_normalized
    
    def load_dataset(self, healthy_folder, diseased_folder, img_size=(64, 64)):
        """
        Load and preprocess images from both folders
        
        Args:
            healthy_folder (str): Path to healthy images folder
            diseased_folder (str): Path to diseased images folder
            img_size (tuple): Size to resize images to
            
        Returns:
            tuple: Preprocessed features and labels
        """
        images = []
        labels = []
        
        # Supported image formats
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        # Load healthy images (label 0)
        print(f"Loading healthy images from {healthy_folder}...")
        for filename in os.listdir(healthy_folder):
            if filename.lower().endswith(supported_formats):
                img_path = os.path.join(healthy_folder, filename)
                try:
                    img_normalized = self.preprocess_image(img_path, img_size)
                    images.append(img_normalized)
                    labels.append(0)  # Healthy label
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
        
        # Load diseased images (label 1)
        print(f"Loading diseased images from {diseased_folder}...")
        for filename in os.listdir(diseased_folder):
            if filename.lower().endswith(supported_formats):
                img_path = os.path.join(diseased_folder, filename)
                try:
                    img_normalized = self.preprocess_image(img_path, img_size)
                    images.append(img_normalized)
                    labels.append(1)  # Diseased label
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images: {np.sum(y == 0)} healthy, {np.sum(y == 1)} diseased")
        return X, y
    
    def prepare_data(self, X, y):
        """
        Apply PCA to reduce dimensionality to match number of qubits
        
        Args:
            X (np.array): Input features
            y (np.array): Labels
            
        Returns:
            tuple: Transformed features and labels
        """
        # Apply PCA for dimensionality reduction
        n_components = min(self.n_qubits, X.shape[1])
        self.pca = PCA(n_components=n_components)
        X_transformed = self.pca.fit_transform(X)
        
        print(f"Reduced features from {X.shape[1]} to {n_components} components")
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        return X_transformed, y
    
    def create_vqc_model(self):
        """
        Create the improved Variational Quantum Classifier model
        
        Returns:
            VQC: Configured VQC model
        """
        # Create feature map with higher order interactions
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits, 
            reps=3,  # Increased repetitions
            entanglement='full',
            insert_barriers=True
        )
        
        # Create more expressive ansatz
        ansatz = RealAmplitudes(
            self.n_qubits, 
            reps=4,  # Increased repetitions
            entanglement='full',
            insert_barriers=True
        )
        
        # Create VQC model without specifying optimizer (use default)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            warm_start=True,
            callback=self._callback  # Add callback for monitoring
        )
        
        return vqc
    
    def _callback(self, weights, obj_func_eval):
        """Callback function to monitor training progress"""
        if not hasattr(self, '_eval_count'):
            self._eval_count = 0
        self._eval_count += 1
        if self._eval_count % 50 == 0:
            print(f"Optimization step {self._eval_count}, Objective function value: {obj_func_eval:.4f}")
    
    def train(self, healthy_folder, diseased_folder):
        """
        Train the improved VQC model on the dataset
        
        Args:
            healthy_folder (str): Path to healthy images folder
            diseased_folder (str): Path to diseased images folder
        """
        # Load dataset (using all available images for better accuracy)
        X, y = self.load_dataset(healthy_folder, diseased_folder)
        
        # Prepare data with PCA
        X_transformed, y = self.prepare_data(X, y)
        
        # Split data with better stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert to numpy arrays
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.int32)
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.int32)
        
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # Store training data for quick retraining
        self.X_train = X_train
        self.y_train = y_train
        
        # Create and train VQC model
        print("Training improved VQC model...")
        print("This may take several minutes. Please be patient...")
        self.model = self.create_vqc_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        self.is_trained = True
        print("Model training completed successfully!")
        return train_accuracy, test_accuracy
    
    def save_model(self, model_path="plant_health_model_vqc.pkl"):
        """
        Save the trained model components to a file
        
        Args:
            model_path (str): Path to save the model file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create a dictionary with all necessary components
        model_data = {
            'pca': self.pca,
            'n_qubits': self.n_qubits,
            'is_trained': self.is_trained,
            'X_train': self.X_train,
            'y_train': self.y_train
        }
        
        # Save to file
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model components saved to {model_path}")

    def load_model(self, model_path="plant_health_model_vqc.pkl"):
        """
        Load model components from a file and retrain the model
        
        Args:
            model_path (str): Path to the model file
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load from file
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore components
        self.pca = model_data['pca']
        self.n_qubits = model_data['n_qubits']
        self.is_trained = model_data['is_trained']
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']
        
        # Recreate and retrain the VQC model
        print("Retraining model from saved data...")
        self.model = self.create_vqc_model()
        self.model.fit(self.X_train, self.y_train)
        
        print(f"Model loaded and retrained from {model_path}")

    def predict(self, image_path):
        """
        Classify a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (prediction, confidence) where prediction is 0/1 and confidence is probability
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            # Preprocess image
            img_normalized = self.preprocess_image(image_path)
            
            # Apply PCA transformation
            if self.pca is None:
                raise ValueError("PCA model not initialized")
            
            img_transformed = self.pca.transform(img_normalized.reshape(1, -1))
            
            # Get prediction
            prediction_result = self.model.predict(img_transformed)
            
            # Handle both array and scalar results safely
            try:
                # Try to get the first element if it's array-like
                prediction = prediction_result[0]
            except (TypeError, IndexError):
                # If that fails, use the result directly (scalar)
                prediction = prediction_result
            
            # For confidence, we'll use a simple approach since predict_proba might not be available
            # We'll return 0.5 as a placeholder confidence
            confidence = 0.5
            
            return int(prediction), float(confidence)
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")
    
    def get_prediction_label(self, prediction):
        """
        Convert prediction number to label
        
        Args:
            prediction (int): 0 or 1
            
        Returns:
            str: "Healthy" or "Diseased"
        """
        return "Healthy" if prediction == 0 else "Diseased"