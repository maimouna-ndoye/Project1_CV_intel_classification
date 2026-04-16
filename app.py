import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf

app = Flask(__name__)

CLASSES     = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_ICONS = {'buildings':'🏢','forest':'🌲','glacier':'🧊',
               'mountain':'⛰️','sea':'🌊','street':'🛣️'}
CLASS_FACTS = {
    'buildings': 'Over 50% of the world population lives in urban areas.',
    'forest':    'Forests cover about 31% of the Earth\'s land area.',
    'glacier':   'Glaciers store about 69% of the world\'s fresh water.',
    'mountain':  'Mountains cover 27% of Earth\'s land surface.',
    'sea':       'The ocean covers more than 70% of the Earth\'s surface.',
    'street':    'There are over 40 million miles of roads worldwide.'
}
IMG_SIZE = 224
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IntelCNN_PyTorch(nn.Module):
    def __init__(self, num_classes=6):
        super(IntelCNN_PyTorch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))

print('Loading PyTorch model...')
pytorch_model = IntelCNN_PyTorch(num_classes=6).to(device)
pytorch_model.load_state_dict(torch.load('maimouna_model.pth', map_location=device))
pytorch_model.eval()
print('✅ PyTorch model loaded')

print('Loading TensorFlow model...')
tf_model = tf.keras.models.load_model('maimouna_model.keras')
print('✅ TensorFlow model loaded')

def preprocess_pytorch(image):
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return t(image).unsqueeze(0).to(device)

def preprocess_tensorflow(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file         = request.files['image']
    image        = Image.open(io.BytesIO(file.read())).convert('RGB')
    model_choice = request.form.get('model', 'pytorch')

    if model_choice == 'pytorch':
        tensor = preprocess_pytorch(image)
        with torch.no_grad():
            outputs = pytorch_model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            idx     = probs.argmax().item()
        all_probs   = [round(p * 100, 1) for p in probs.tolist()]
        confidence  = round(probs[idx].item() * 100, 1)
    else:
        arr        = preprocess_tensorflow(image)
        probs      = tf_model.predict(arr, verbose=0)[0]
        idx        = int(np.argmax(probs))
        all_probs  = [round(float(p) * 100, 1) for p in probs]
        confidence = round(float(probs[idx]) * 100, 1)

    predicted = CLASSES[idx]
    return jsonify({
        'class':      predicted,
        'icon':       CLASS_ICONS[predicted],
        'confidence': confidence,
        'fact':       CLASS_FACTS[predicted],
        'all_probs':  dict(zip(CLASSES, all_probs)),
        'model':      model_choice
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)