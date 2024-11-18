from flask import Flask, render_template, request, send_from_directory, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import time
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Hide all GPUs
    tf.config.set_visible_devices([], 'GPU')
# Define upload and result folders, and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

# Ensure the directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load pre-trained model
model = load_model('checkpoint.keras')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to preprocess the image for model input
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize image to match model input size
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure files are uploaded
        if 'files' not in request.files:
            return 'No files uploaded'
        files = request.files.getlist('files')
        
        # Debug: Print the list of uploaded files
        print(f"Uploaded files: {[file.filename for file in files]}")
        
        # Process each uploaded file
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                # Add a unique identifier to filename
                unique_filename = str(int(time.time())) + "_" + file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                # Open the uploaded image
                img = Image.open(filepath)

                # Preprocess the image
                processed_img = preprocess_image(img)

                # Perform segmentation (model inference)
                segmented_output = model.predict(processed_img)  # Assuming output shape (1, 256, 256, 1)

                # Post-process the segmentation output
                segmented_output = (segmented_output > 0.5).astype(np.uint8)  # Binarize output (thresholding)
                segmented_output = segmented_output[0, :, :, 0]  # Remove batch and channel dimension
                segmented_output = cv2.resize(segmented_output, (img.width, img.height))

                # Create overlay image
                original_img = np.array(img)
                mask_rgb = cv2.cvtColor(segmented_output * 255, cv2.COLOR_GRAY2RGB)
                overlay = cv2.addWeighted(original_img, 0.6, mask_rgb, 0.4, 0)

                # Save images
                original_path = os.path.join(app.config['RESULT_FOLDER'], 'original_' + unique_filename)
                mask_path = os.path.join(app.config['RESULT_FOLDER'], 'mask_' + unique_filename)
                overlay_path = os.path.join(app.config['RESULT_FOLDER'], 'overlay_' + unique_filename)

                Image.fromarray(original_img).save(original_path)
                Image.fromarray(segmented_output * 255).save(mask_path)
                Image.fromarray(overlay).save(overlay_path)

                # Append result paths
                results.append({
                    'original': 'original_' + unique_filename,
                    'mask': 'mask_' + unique_filename,
                    'overlay': 'overlay_' + unique_filename
                })

        # Render results
        return render_template('result.html', results=results)

    return render_template('index.html')


# Route to serve result images
@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

