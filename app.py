from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', result=None, image_url=None)

@app.route('/predict', methods=['POST'])
def image_predict():
    # Get the uploaded file
    file = request.files['image']

    # Check if the file exists
    if not file:
        return "No file uploaded."

    # Open the image file
    img = Image.open(file.stream)  # Read image from the file stream

    # Preprocess the image
    img = img.resize((256, 256))  # Resize to the input shape expected by the model
    img_array = np.array(img)  # Convert to NumPy array

    # Normalize the image array
    img_array = img_array.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

    # Reshape for the model input
    test_input = img_array.reshape((1, 256, 256, 3))

    # Make prediction
    result = model.predict(test_input)

    # Assuming result is a binary classification (0 or 1)
    prediction = 'It is a Dog.' if result[0][0] > 0.5 else 'It is a Cat.'

    # Save the uploaded image temporarily to display it
    image_url = os.path.join('static', 'uploads', file.filename)
    img.save(image_url)  # Save the image to the static folder

    return render_template('index.html', result=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
