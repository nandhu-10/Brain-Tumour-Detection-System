Brain Tumor Segmentation using MONAI and Flask
This project is a web-based application that uses a deep learning model to detect and segment brain tumors from MRI scans.

Key Features:

• MONAI Framework: Utilizes the MONAI library, specifically designed for medical imaging AI.

• Deep Learning Model: Employs a UNet architecture for precise image segmentation.

• Flask Web App: A user-friendly interface to upload MRI images and view detection results.

• Real-time Visualization: Displays the original scan alongside the predicted tumor area (overlay).

Tech Stack:

• Backend: Python, Flask

• AI/ML: PyTorch, MONAI, Torchvision

• Image Processing: Pillow, NumPy, Matplotlib

• Frontend: HTML, CSS (Jinja2 templates)

How to Run:

1. Install dependencies: `pip install -r requirements.txt`

2. Run the application: `python app.py`

3. Upload an MRI image through the browser to see the prediction.
