import os
import pickle
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA


class PlantHealthClassifier:
    """
    Variational Quantum Classifier (VQC) for Plant Leaf Disease Classification.
    
    Architecture:
      - Input resolution: 32x32 RGB (3072 features, normalized [0, 1])
      - Dimensionality reduction: PCA (4 principal components)
      - Quantum feature encoding: ZZFeatureMap (4 qubits, reps=1, linear entanglement)
      - Quantum ansatz: RealAmplitudes (4 qubits, reps=2, linear entanglement)
      - Classical optimizer: COBYLA (maxiter=100)
    """

    def __init__(self, n_qubits=4):
        """
        Initialize the Plant Health Classifier.

        Args:
            n_qubits (int): Number of qubits and corresponding PCA components (default: 4).
        """
        self.n_qubits = n_qubits
        self.pca = None
        self.model = None
        self.is_trained = False
        self.X_train = None
        self.y_train = None

    def preprocess_image(self, image_path, img_size=(32, 32)):
        """
        Preprocess a single image for classification.

        Args:
            image_path (str): Path to the image file.
            img_size (tuple): Target resize dimensions (width, height).

        Returns:
            np.ndarray: 1D normalized float array of pixel values in [0, 1].
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(img_size)

        img_array = np.array(img)
        img_flat = img_array.flatten()
        img_normalized = img_flat.astype('float32') / 255.0

        return img_normalized

    def load_dataset(self, healthy_folder, diseased_folder, img_size=(32, 32), max_samples=50):
        """
        Load and preprocess images from healthy and diseased folders.

        Args:
            healthy_folder (str): Directory containing healthy leaf images.
            diseased_folder (str): Directory containing diseased leaf images.
            img_size (tuple): Resize dimensions.
            max_samples (int): Maximum number of samples to load per class.

        Returns:
            tuple: (X, y) feature and label numpy arrays.
        """
        images = []
        labels = []
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        # Load healthy samples (class 0)
        healthy_count = 0
        if os.path.exists(healthy_folder):
            for filename in sorted(os.listdir(healthy_folder)):
                if healthy_count >= max_samples:
                    break
                if filename.lower().endswith(supported_formats):
                    img_path = os.path.join(healthy_folder, filename)
                    try:
                        img_normalized = self.preprocess_image(img_path, img_size)
                        images.append(img_normalized)
                        labels.append(0)
                        healthy_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to load {filename}: {e}")

        # Load diseased samples (class 1)
        diseased_count = 0
        if os.path.exists(diseased_folder):
            for filename in sorted(os.listdir(diseased_folder)):
                if diseased_count >= max_samples:
                    break
                if filename.lower().endswith(supported_formats):
                    img_path = os.path.join(diseased_folder, filename)
                    try:
                        img_normalized = self.preprocess_image(img_path, img_size)
                        images.append(img_normalized)
                        labels.append(1)
                        diseased_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to load {filename}: {e}")

        X = np.array(images, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        return X, y

    def prepare_data(self, X, y):
        """
        Fit PCA to reduce feature dimensionality to match the number of qubits.

        Args:
            X (np.ndarray): High-dimensional input feature vectors.
            y (np.ndarray): Target labels.

        Returns:
            tuple: (X_transformed, y) where X_transformed has shape (N, n_qubits).
        """
        n_components = min(self.n_qubits, X.shape[1])
        self.pca = PCA(n_components=n_components)
        X_transformed = self.pca.fit_transform(X)
        return X_transformed, y

    def create_vqc_model(self):
        """
        Construct and configure the Variational Quantum Classifier.

        Returns:
            VQC: Configured Qiskit VQC instance.
        """
        # Quantum feature encoding using ZZFeatureMap
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=1,
            entanglement='linear',
            insert_barriers=False
        )

        # Variational ansatz circuit using RealAmplitudes
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=2,
            entanglement='linear',
            insert_barriers=False
        )

        # Classical optimizer
        optimizer = COBYLA(maxiter=100)

        # Variational Quantum Classifier
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            warm_start=True
        )

        return vqc

    def train(self, healthy_folder, diseased_folder, test_size=0.2, random_state=42, max_samples=50):
        """
        Train the VQC model on dataset images with stratified train/test split.

        Args:
            healthy_folder (str): Path to healthy image folder.
            diseased_folder (str): Path to diseased image folder.
            test_size (float): Fraction of data to use for test evaluation.
            random_state (int): Seed for deterministic data splitting.
            max_samples (int): Max images per class.

        Returns:
            tuple: (train_accuracy, test_accuracy)
        """
        X, y = self.load_dataset(healthy_folder, diseased_folder, max_samples=max_samples)
        if len(X) == 0:
            raise ValueError("No valid images found in the specified dataset directories.")

        X_transformed, y = self.prepare_data(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.X_train = np.array(X_train, dtype=np.float64)
        self.y_train = np.array(y_train, dtype=np.int32)
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.int32)

        self.model = self.create_vqc_model()
        self.model.fit(self.X_train, self.y_train)

        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(X_test)

        train_accuracy = float(accuracy_score(self.y_train, y_pred_train))
        test_accuracy = float(accuracy_score(y_test, y_pred_test))

        self.is_trained = True
        return train_accuracy, test_accuracy

    def save_model(self, model_path="plant_health_model.pkl"):
        """
        Save the fitted PCA transformer, training state, and parameters to a file.

        Args:
            model_path (str): File destination path.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        model_data = {
            'pca': self.pca,
            'n_qubits': self.n_qubits,
            'is_trained': self.is_trained,
            'X_train': self.X_train,
            'y_train': self.y_train
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, model_path="plant_health_model.pkl"):
        """
        Load model state from file and initialize the VQC classifier.

        Args:
            model_path (str): Path to the saved pickle model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.pca = model_data['pca']
        self.n_qubits = model_data['n_qubits']
        self.is_trained = model_data['is_trained']
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']

        self.model = self.create_vqc_model()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, image_path):
        """
        Perform inference on a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: (prediction_int, confidence_float) where 0=Healthy, 1=Diseased.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be loaded or trained before inference.")

        img_normalized = self.preprocess_image(image_path)
        img_transformed = self.pca.transform(img_normalized.reshape(1, -1))

        prediction_result = self.model.predict(img_transformed)
        try:
            prediction = int(prediction_result[0])
        except (TypeError, IndexError):
            prediction = int(prediction_result)

        confidence = 0.5
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(img_transformed)
                if len(proba.shape) == 2 and proba.shape[1] > prediction:
                    confidence = float(proba[0][prediction])
                elif len(proba.shape) == 1 and len(proba) > prediction:
                    confidence = float(proba[prediction])
        except Exception:
            confidence = 0.5

        return prediction, confidence

    def get_prediction_label(self, prediction):
        """
        Convert numeric prediction (0/1) to human-readable label.

        Args:
            prediction (int): 0 or 1.

        Returns:
            str: "Healthy" or "Diseased".
        """
        return "Healthy" if prediction == 0 else "Diseased"
