import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from vqc_model import PlantHealthClassifier

app = Flask(__name__)

# Application configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB maximum upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

MODEL_PATH = "plant_health_model.pkl"

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize and load model
classifier = None
model_loaded = False


def allowed_file(filename):
    """Check if uploaded file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_model():
    """Load the pre-trained VQC model from disk."""
    global classifier, model_loaded
    try:
        if os.path.exists(MODEL_PATH):
            classifier = PlantHealthClassifier(n_qubits=4)
            classifier.load_model(MODEL_PATH)
            model_loaded = True
            print(f"[*] VQC model successfully loaded from {MODEL_PATH}")
            return True
        else:
            print(f"[!] Model file '{MODEL_PATH}' not found. Run 'python train_model.py' first.")
            return False
    except Exception as e:
        print(f"[!] Error loading model: {str(e)}")
        model_loaded = False
        return False


# Attempt model load on startup
initialize_model()


@app.route('/')
def index():
    """Render main application interface."""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/model_info', methods=['GET'])
def model_info():
    """Return runtime information about the loaded VQC model."""
    return jsonify({
        'status': 'ready' if model_loaded else 'not_loaded',
        'model_type': 'Variational Quantum Classifier (VQC)',
        'qubits': 4,
        'feature_map': 'ZZFeatureMap (reps=1, linear)',
        'ansatz': 'RealAmplitudes (reps=2, linear)',
        'optimizer': 'COBYLA (maxiter=100)'
    })


@app.route('/classify', methods=['POST'])
def classify():
    """Handle image upload, execute VQC inference, and return prediction."""
    global classifier, model_loaded

    if not model_loaded or classifier is None:
        # Try loading once more if not loaded yet
        if not initialize_model():
            return jsonify({
                'status': 'error',
                'message': 'Model is not loaded. Please run train_model.py first.'
            }), 503

    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image file provided in request.'
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected for upload.'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'Invalid file format. Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    filename = secure_filename(file.filename) or "upload.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # Run inference through the quantum pipeline
        prediction, confidence = classifier.predict(filepath)
        label = classifier.get_prediction_label(prediction)

        return jsonify({
            'status': 'success',
            'prediction': label,
            'confidence': confidence,
            'prediction_code': int(prediction)
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Inference error: {str(e)}'
        }), 500

    finally:
        # Guarantee cleanup of uploaded temporary image
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
