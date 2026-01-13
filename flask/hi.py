import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Optional: set log level to hide TensorFlow INFO/WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -------- CONFIG --------
MODEL_PATH = "D:\D\cnn_fruit_model.h5"  # Change to your .h5 file path
LABELS_PATH = "D:\D\class_labels.txt"  # Optional: path to labels file, one per line
IMAGE_PATH = "image.jpg"  # Change to your image path
INPUT_SIZE = (224, 224)

# -------- LOAD MODEL AND LABELS --------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def load_labels(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return None


class_names = load_labels(LABELS_PATH)


# -------- PREDICT FUNCTION --------
def predict_image(model, img_path, input_size=(128, 128), class_names=None):
    img = load_img(img_path, target_size=input_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
    arr = arr.reshape((1, np.prod(input_size) * 3))  # shape: (1, 57600) for 160x120x3
    preds = model.predict(arr)
    probs = tf.nn.softmax(preds[0]).numpy()
    top = int(np.argmax(probs))
    confidence = float(probs[top])
    label = (
        class_names[top] if class_names and 0 <= top < len(class_names) else str(top)
    )
    return label, confidence


# -------- MAIN USAGE --------
if __name__ == "__main__":
    # Optionally: allow arg passing for image path
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    label, conf = predict_image(model, IMAGE_PATH, INPUT_SIZE, class_names)
    print(f"Predicted label: {label}")
    print(f"Confidence:     {conf:.4f}")
