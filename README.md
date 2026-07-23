# 🌿 Quantum Plant Health Classifier

A **Quantum Machine Learning** web application that classifies plant leaf images as **Healthy** or **Diseased** using a **Variational Quantum Classifier (VQC)** built with **Qiskit**. The application provides an intuitive Flask-based interface for uploading images and receiving real-time predictions with confidence scores.

> Developed as an exploration of Quantum Machine Learning (QML) by combining quantum circuits with classical image preprocessing for plant disease detection.

---

## 📸 Demo

> **Live Demo:** *(Add deployment link if available)*

### Screenshots

| Home Page          | Prediction Result  |
| ------------------ | ------------------ |
| *(Add Screenshot)* | *(Add Screenshot)* |

---

## ✨ Features

* 🌱 Plant leaf disease classification
* ⚛️ Quantum Machine Learning using Qiskit VQC
* 📷 Image upload through a Flask web interface
* 📊 Confidence score for every prediction
* 🖥️ Clean and responsive user interface
* 🧠 PCA-based feature reduction before quantum encoding
* 💾 Pre-trained model loading for faster inference

---

## 🛠️ Tech Stack

| Category             | Technologies                    |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Backend              | Flask                           |
| Quantum Computing    | Qiskit, Qiskit Machine Learning |
| Machine Learning     | Scikit-learn                    |
| Image Processing     | Pillow                          |
| Scientific Computing | NumPy                           |
| Visualization        | Matplotlib                      |

---

## 📂 Project Structure

```text
AIproject/
│
├── app.py                     # Flask application
├── vqc_model.py               # Quantum classifier implementation
├── train_model.py             # Model training script
├── plant_health_model.pkl     # Trained model
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── uploads/
│
└── Dataset/
    ├── Tomato_Healthy/
    └── Tomato_Diseased/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/harshit-s11/AIproject.git
cd AIproject
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```text
http://localhost:5000
```

Upload a plant leaf image and receive a prediction indicating whether the leaf is **Healthy** or **Diseased**, along with the model's confidence score.

---

## 🧠 Model Workflow

```text
Plant Leaf Image
        │
        ▼
 Image Preprocessing
        │
        ▼
 Feature Extraction
        │
        ▼
 PCA Dimensionality Reduction
        │
        ▼
 Quantum Feature Encoding
        │
        ▼
 Variational Quantum Circuit (VQC)
        │
        ▼
 Prediction
        │
        ▼
Healthy / Diseased
```

---

## 🔬 Quantum Machine Learning Pipeline

The classification process consists of the following stages:

1. Image preprocessing and normalization
2. Feature reduction using Principal Component Analysis (PCA)
3. Quantum state encoding using **ZZFeatureMap**
4. Training a **Variational Quantum Classifier (VQC)** with **RealAmplitudes**
5. Optimization using the **COBYLA** optimizer
6. Prediction with confidence estimation

---

## 📦 Dependencies

Major libraries used in this project include:

* Qiskit
* Qiskit Machine Learning
* Flask
* NumPy
* Scikit-learn
* Pillow
* Matplotlib

For the complete list, see:

```text
requirements.txt
```

---

## 🚀 Future Improvements

* Support multiple crop species
* Multi-class disease classification
* Deploy on cloud platforms
* Mobile-friendly interface
* Improved quantum circuit optimization
* Explainable AI visualizations
* Larger training datasets
* Integration with real quantum hardware

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Harshit Sharma**

B.Tech Computer Science Engineering

VIT Chennai

GitHub: https://github.com/harshit-s11
