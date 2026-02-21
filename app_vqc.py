from flask import Flask, render_template, request, jsonify
import os
from vqc_improved import ImprovedPlantHealthClassifier
from vqc_fast import FastPlantHealthClassifier
from vqc_old import OldPlantHealthClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global classifier instances
improved_classifier = None
fast_classifier = None
old_classifier = None
model_trained = False
active_model = None
active_model_type = None

# Model paths
improved_model_path = "plant_health_model_vqc.pkl"
fast_model_path = "plant_health_model_fast.pkl"
old_model_path = "plant_health_model_old.pkl"

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def load_pretrained_model():
    """Load pre-trained VQC model (prioritizing fast model, then old, then improved)"""
    global improved_classifier, fast_classifier, poor_classifier, model_trained, active_model, active_model_type
    
    try:
        # Try to load fast model first (it's faster)
        if os.path.exists(fast_model_path):
            # Initialize fast classifier
            fast_classifier = FastPlantHealthClassifier(n_qubits=4)
            
            # Load the trained model
            fast_classifier.load_model(fast_model_path)
            active_model = fast_classifier
            active_model_type = "fast"
            model_trained = True
            print(f"Fast VQC model loaded from {fast_model_path}")
            return True
        elif os.path.exists(old_model_path):
            # Try old model next
            old_classifier = OldPlantHealthClassifier(n_qubits=2)
            
            # Load the trained model
            old_classifier.load_model(old_model_path)
            active_model = old_classifier
            active_model_type = "old"
            model_trained = True
            print(f"Old VQC model loaded from {old_model_path}")
            return True
        elif os.path.exists(improved_model_path):
            # Fallback to improved model
            improved_classifier = ImprovedPlantHealthClassifier(n_qubits=6)
            
            # Load the trained model
            improved_classifier.load_model(improved_model_path)
            active_model = improved_classifier
            active_model_type = "improved"
            model_trained = True
            print(f"Improved VQC model loaded from {improved_model_path}")
            return True
        else:
            print("No pre-trained VQC model found. Please run one of the training scripts first.")
            return False
    except Exception as e:
        print(f"Error loading pre-trained VQC model: {str(e)}")
        return False

# Load model when app starts
load_pretrained_model()

@app.route('/')
def index():
    """Render the main page"""
    global model_trained, active_model_type
    model_type = active_model_type if active_model_type else "none"
    return render_template('index.html', model_loaded=model_trained, model_type=model_type)

@app.route('/model_info')
def model_info():
    """Return information about the loaded model"""
    global active_model_type
    return jsonify({
        'model_type': active_model_type if active_model_type else 'none'
    })

@app.route('/classify', methods=['POST'])
def classify():
    """Classify an uploaded image"""
    global active_model, model_trained, active_model_type
    
    # Check if model is trained
    if not model_trained or active_model is None:
        return jsonify({
            'status': 'error',
            'message': 'VQC model not loaded. Please run a training script first to train a model.'
        })
    
    # Check if file is present in request
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image file provided'
        })
    
    file = request.files['image']
    
    # Check if file has a filename
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No image selected'
        })
    
    # Initialize filepath variable
    filepath = None
    
    try:
        # Save uploaded file
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Ensure filename is a string
        filename = str(file.filename) if file.filename else 'uploaded_image.jpg'
        filepath = os.path.abspath(os.path.join(upload_folder, filename))
        file.save(filepath)
        
        # Classify the image
        prediction, confidence = active_model.predict(filepath)
        label = active_model.get_prediction_label(prediction)
        
        # Remove uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Return result with model type info
        return jsonify({
            'status': 'success',
            'prediction': label,
            'confidence': confidence,
            'prediction_code': int(prediction),
            'model_type': active_model_type
        })
        
    except Exception as e:
        # Clean up file if it exists
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'status': 'error',
            'message': f'Error classifying image: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port