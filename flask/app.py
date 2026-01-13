# from flask import Flask, render_template, request
# import torch
# from torchvision import transforms
# from PIL import Image
# from io import BytesIO

# # Model and config
# MODEL_PATH = "D:\\D\\model.pth"      # Change to your model .pth file
# LABELS_PATH = "D:\\D\\class_labels.txt"
# INPUT_SIZE = (128, 128)

# import torch.nn as nn

# class MyCNNModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Define layers just like in training script
#         # Example:
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.pool  = nn.MaxPool2d(2)
#         self.fc1   = nn.Linear(32 * 64 * 64, 10)  # adjust for your architecture

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         # (Flatten, more layers...)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyCNNModel()
# state_dict = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(state_dict)
# model.eval()


# # Load PyTorch model
# # model = torch.load(MODEL_PATH, map_location=device)
# # model.eval()
# #
# def load_labels(path):
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return [ln.strip() for ln in f if ln.strip()]
#     except Exception:
#         return None

# class_names = load_labels(LABELS_PATH)

# def predict_image(model, img_bytes, input_size=(128,128), class_names=None):
#     # PIL image, preprocess
#     img = Image.open(BytesIO(img_bytes)).convert("RGB")
#     preprocess = transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)   # adjust as per model training!
#     ])
#     arr = preprocess(img).unsqueeze(0).to(device)  # shape: [1, 3, H, W]
#     with torch.no_grad():
#         logits = model(arr)
#         # Multi-class
#         if logits.shape[-1] > 1:
#             probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
#             top = int(probs.argmax())
#             confidence = float(probs[top])
#             label = class_names[top] if class_names and 0 <= top < len(class_names) else str(top)
#         else:
#             prob = float(torch.sigmoid(logits).cpu().numpy()[0][0])
#             label = "Rotten" if prob > 0.5 else "Fresh"    # Edit as needed
#             if class_names is not None and len(class_names) > 1:
#                 label = class_names[1] if prob > 0.5 else class_names[0]
#             confidence = prob if prob > 0.5 else 1.0 - prob
#     return label, confidence

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     label = confidence = annotated_image_url = None
#     if request.method == "POST":
#         file = request.files.get("image")
#         if file and file.filename:
#             img_bytes = file.read()
#             label, confidence = predict_image(model, img_bytes, INPUT_SIZE, class_names)
#     return render_template("index.html", label=label, confidence=confidence, annotated_image_url=annotated_image_url)

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
import re

# Load model/labels as before
MODEL_PATH = "D:\\D\\hybrid_model.h5"
LABELS_PATH = "D:\\D\\classnames.txt"
INPUT_SIZE = (224, 224)
TF_ENABLE_ONEDNN_OPTS = 0

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def load_labels(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return None


class_names = load_labels(LABELS_PATH)


def predict_image(model, img_bytes, input_size=(224, 224), class_names=None):
    img = load_img(BytesIO(img_bytes), target_size=input_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)
    preds = model.predict(arr)
    if preds.shape[-1] > 1:
        probs = tf.nn.softmax(preds[0]).numpy()
        top = int(np.argmax(probs))
        confidence = round(float(probs[top]) * 1000, 2)
        print(confidence)
        if confidence > 50.00:
            label = (
                class_names[top]
                if class_names and 0 <= top < len(class_names)
                else str(top)
            )
            label = label.replace("_", " ")
        else:
            label = "not identified"
            confidence = 0
    else:
        confidence = round(float(probs[top]) * 1000, 2)
        label = class_names[1] if confidence > 0.8 else print("not identified")
        text = re.sub(r"_+", " ", text)
        words = text.split()
        if len(words) >= 2:
            words[0], words[1] = words[1], words[0]
        label = " ".join(words)
        confidence = confidence if confidence > 0.5 else 0
    return label, confidence


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    label = confidence = annotated_image_url = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            img_bytes = file.read()
            label, confidence = predict_image(model, img_bytes, INPUT_SIZE, class_names)

    return render_template(
        "HTML.html",
        label=label,
        confidence=confidence,
        annotated_image_url=annotated_image_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
