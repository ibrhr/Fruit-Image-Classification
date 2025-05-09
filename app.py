import torch
from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
from inference import InferenceModel

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# instantiate once
inferer = InferenceModel(
    checkpoint_path='checkpoints/best_model.pth',
    class_map_path='class_to_idx.json',
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify(error="No file"),400
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    cls, conf = inferer.predict(img)
    return jsonify(prediction=cls, confidence=round(conf*100,2))

if __name__=='__main__':
    app.run(debug=False, host='0.0.0.0')
