import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import monai
import numpy as np
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Configure upload and result folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Define the model path (use environment variable or default)

model_path = os.path.join(os.path.dirname(__file__), 'monai', 'model.pt')

# Load your pre-trained MONAI model
try:
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        kernel_size=3,
        act=("LEAKYRELU", {"negative_slope": 0.01}),
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Image processing function
def process_image(image_path):
    try:
        image = Image.open(image_path).convert('L')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Prediction function
def predict_and_visualize(image_tensor, original_image):
    try:
        with torch.no_grad():
            output = model(image_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        threshold = 0.5
        binary_mask = (prediction > threshold).astype(np.uint8)

        # Create overlay using Pillow
        overlay = original_image.convert("RGB")
        draw = ImageDraw.Draw(overlay)

        y, x = np.where(binary_mask == 1)

        if len(x)>0 and len(y)>0:
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=(0, 0, 255), width=2)
            tumor_detected = True
        else:
            tumor_detected = False

        return np.array(overlay), tumor_detected, prediction

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():


    if request.method == 'POST':
        print(request.files)  # Debug print
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        print(file.filename)  # Debug print
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            original_image = Image.open(filename).convert('L')
            image_tensor = process_image(filename)
            if image_tensor is None:
                return render_template('index.html', error="Error processing image")

            result_overlay, tumor_detected, prediction = predict_and_visualize(image_tensor, original_image)

            if result_overlay is None or tumor_detected is None or prediction is None:
                return render_template('index.html', error="Error during prediction")

            overlay_filename = os.path.join(app.config['RESULT_FOLDER'], 'overlay.png')
            Image.fromarray(result_overlay).save(overlay_filename)

            prediction_filename = os.path.join(app.config['RESULT_FOLDER'], 'prediction.png')
            plt.imsave(prediction_filename, prediction, cmap='gray')

            return render_template('result.html',
                                   overlay_image=os.path.join('static', 'results', 'overlay.png'),
                                   prediction_image=os.path.join('static', 'results', 'prediction.png'),
                                   tumor_detected=tumor_detected)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return render_template('index.html', error=f"An error occurred: {e}")

    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)