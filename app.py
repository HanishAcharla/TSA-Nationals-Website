import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, session
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = 'your_secret_key_here'  # Replace with a secure key for production

# Load your trained model once when the server starts
model_path = "/Users/hanishacharla/Downloads/ResNet-Model.h5"   # Adjust this path as needed
model = tf.keras.models.load_model(model_path)
class_names = ['Complex', 'Frog_Eye_Leaf_Spot', 'Healthy', 'Powdery_Mildew', 'Rust', 'Scab']

# Disease descriptions and recommendations
disease_info = {
    'Complex': {
        'description': 'Multiple diseases may be present on this leaf.',
        'recommendations': 'Consult with a specialist for accurate diagnosis and treatment options.'
    },
    'Frog_Eye_Leaf_Spot': {
        'description': 'Frog Eye Leaf Spot is caused by the fungus Botryosphaeria obtusa. It creates circular lesions with brown or purple margins.',
        'recommendations': 'Apply fungicide treatments specifically designed for frog eye leaf spot. Ensure proper spacing between trees for adequate airflow.'
    },
    'Healthy': {
        'description': 'This leaf appears to be healthy with no visible disease symptoms.',
        'recommendations': 'Continue with regular maintenance practices including proper watering, fertilization, and preventive sprays.'
    },
    'Powdery_Mildew': {
        'description': 'Powdery Mildew appears as a white powdery substance on leaves caused by fungal pathogens.',
        'recommendations': 'Apply fungicides with sulfur or potassium bicarbonate. Improve air circulation around trees and avoid overhead watering.'
    },
    'Rust': {
        'description': 'Apple Rust is caused by fungi in the Gymnosporangium genus, creating orange or yellow spots on leaves.',
        'recommendations': 'Remove nearby juniper plants (alternate host), apply protective fungicides in spring, and practice good orchard sanitation.'
    },
    'Scab': {
        'description': 'Apple Scab is caused by the fungus Venturia inaequalis, creating olive-green to brown lesions on leaves.',
        'recommendations': 'Apply fungicides early in the growing season. Remove and destroy fallen leaves to reduce fungal spores for next season.'
    }
}

# Function to preprocess the image for ResNet-50
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet-50 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize using ResNet preprocessing
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
        
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
        
    if file:
        # Create a unique filename to prevent overwriting
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(filepath)
        
        # Store the filename in the session for the preview page
        session['uploaded_image'] = unique_filename
        
        return redirect(url_for('preview_image'))
    
    return redirect(url_for('index'))

@app.route('/preview', methods=['GET'])
def preview_image():
    if 'uploaded_image' not in session:
        flash('No image uploaded', 'error')
        return redirect(url_for('index'))
        
    image_file = session['uploaded_image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
    
    # For display in the template
    display_path = f"/static/uploads/{image_file}"
    
    return render_template('preview.html', image_path=display_path)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'uploaded_image' not in session:
        flash('No image to classify', 'error')
        return redirect(url_for('index'))
        
    image_file = session['uploaded_image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
    display_path = f"/static/uploads/{image_file}"
    
    # Preprocess and predict
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    
    # Get prediction probabilities for all classes
    probs = predictions[0].tolist()
    formatted_probs = {class_names[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(probs)}
    
    # Get the top prediction
    predicted_class = np.argmax(predictions, axis=1)[0]
    result = class_names[predicted_class]
    
    # Get disease information
    description = disease_info[result]['description']
    recommendations = disease_info[result]['recommendations']
    
    return render_template('result.html', 
                          result=result, 
                          image_path=display_path,
                          description=description,
                          recommendations=recommendations,
                          probabilities=formatted_probs)

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True)