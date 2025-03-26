import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'your_secret_key_here'  # Replace with a secure key for production

# Load your trained model once when the server starts.
model_path = "/Users/hanishacharla/Downloads/ResNet-Model.h5"   # Adjust this path as needed.
model = tf.keras.models.load_model(model_path)
class_names = ['Complex', 'Frog_Eye_Leaf_Spot', 'Healthy', 'Powdery_Mildew', 'Rust', 'Scab']

# Function to preprocess the image for ResNet-50
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet-50 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize using ResNet preprocessing
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess and predict
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            result = class_names[predicted_class]

            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
