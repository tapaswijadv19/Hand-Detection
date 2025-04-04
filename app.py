from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load trained model
model = tf.keras.models.load_model("isl_detection_model.keras")

# Define class labels
labels = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
label_dict = {i: labels[i] for i in range(len(labels))}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = cv2.resize(image, (64, 64))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/")
def home():
    """Default route to check API status"""
    return jsonify({"message": "ISL Detection API is running successfully!"})

@app.route("/predict/image", methods=["POST"])
def predict_image():
    """Handle image uploads and return predictions"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    img = preprocess_image(image)
    predictions = model.predict(img)[0]
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[predicted_class])

    return jsonify({
        "prediction": label_dict[predicted_class],
        "confidence": confidence
    })

@app.route("/predict/video", methods=["POST"])
def predict_video():
    """Handle video file and return frame-wise predictions"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    video = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if video is None:
        return jsonify({"error": "Invalid video format"}), 400

    cap = cv2.VideoCapture(video)
    predictions_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = preprocess_image(frame)
        predictions = model.predict(img)[0]
        predicted_class = np.argmax(predictions)
        predictions_list.append(label_dict[predicted_class])

    cap.release()

    return jsonify({
        "predictions": predictions_list
    })

@app.route("/predict/webcam", methods=["GET"])
def predict_webcam():
    """Capture webcam feed and return real-time predictions"""
    cap = cv2.VideoCapture(0)  # Use default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess_image(frame)
        predictions = model.predict(img)[0]
        predicted_class = np.argmax(predictions)
        print(f"Predicted: {label_dict[predicted_class]}")  # Display prediction in console

        cv2.putText(frame, label_dict[predicted_class], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ISL Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Webcam stream closed"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Run on port 5000
