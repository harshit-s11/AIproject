<p align="center">
  <img src="banner.png" alt="Quantum Plant Health Classifier Banner" width="100%">
</p>

# 🌿 Quantum Plant Health Classifier

<p align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-4B8BBE?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

A Quantum Machine Learning web application that classifies plant leaf images as **Healthy** or **Diseased** using a **Variational Quantum Classifier (VQC)** built with **Qiskit**.

The project combines classical image preprocessing with quantum machine learning techniques and provides a Flask-based web interface for real-time plant disease prediction with confidence scores.

---

# 📖 Project Overview

Quantum Machine Learning (QML) combines quantum computing principles with machine learning to explore new approaches for solving complex learning problems.

This project applies QML to plant disease classification by integrating image preprocessing, feature reduction using PCA, and a Variational Quantum Classifier (VQC) implemented with Qiskit. The result is an interactive web application capable of classifying plant leaf images as healthy or diseased.

---

# ✨ Features

- Plant leaf disease classification
- Quantum Machine Learning using Qiskit
- Variational Quantum Classifier (VQC)
- Flask-based web application
- Image upload interface
- Confidence score prediction
- PCA-based feature reduction
- Pre-trained model loading
- Responsive user interface

---

# 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Backend | Flask |
| Quantum Computing | Qiskit, Qiskit Machine Learning |
| Machine Learning | Scikit-learn |
| Image Processing | Pillow |
| Scientific Computing | NumPy |
| Visualization | Matplotlib |

---

# ⚛️ Quantum Machine Learning Workflow

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
Variational Quantum Classifier (VQC)
        │
        ▼
Prediction
        │
        ▼
Healthy / Diseased
```

---

# 🔬 Quantum Pipeline

The classification process consists of:

1. Image preprocessing and normalization
2. Feature extraction
3. PCA dimensionality reduction
4. Quantum feature encoding using **ZZFeatureMap**
5. Variational Quantum Classifier using **RealAmplitudes**
6. Optimization using **COBYLA**
7. Prediction with confidence estimation

---

# 📂 Project Structure

```text
AIproject/
│
├── app.py
├── vqc_model.py
├── train_model.py
├── plant_health_model.pkl
├── requirements.txt
├── banner.png
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

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/harshit-s11/AIproject.git
```

---

## Navigate to the Project

```bash
cd AIproject
```

---

## Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://localhost:5000
```

Upload a plant leaf image and receive a prediction indicating whether the leaf is **Healthy** or **Diseased**, along with the model's confidence score.

---

# 📊 Model Components

- Variational Quantum Classifier (VQC)
- ZZFeatureMap
- RealAmplitudes Ansatz
- COBYLA Optimizer
- PCA Feature Reduction
- Classical Image Preprocessing

---

# 💼 Applications

- Smart Agriculture
- Plant Disease Detection
- Precision Farming
- Quantum Machine Learning Research
- Computer Vision
- AI-powered Agricultural Systems

---

# 🎯 Learning Outcomes

Through this project, I gained practical experience with:

- Quantum Machine Learning
- Qiskit
- Variational Quantum Classifiers
- Quantum Feature Encoding
- Flask Development
- Image Processing
- PCA Dimensionality Reduction
- Hybrid Classical-Quantum AI Systems

---

# 🔮 Future Improvements

- Multi-class disease classification
- Additional crop species support
- Cloud deployment
- Mobile-responsive interface
- Explainable AI visualizations
- Quantum circuit optimization
- Real quantum hardware integration
- REST API support

---

# 📜 License

This project is licensed for educational and research purposes.

---

# 👨‍💻 Author

**Harshit Sharma**

Final-year B.Tech Computer Science Engineering Student

VIT Chennai

🔗 GitHub: https://github.com/harshit-s11

🔗 LinkedIn: https://linkedin.com/in/harshit-sharma24

---

⭐ If you found this repository useful, consider giving it a star.
