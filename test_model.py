import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("isl_finetuned_model.keras")

# Get model input size dynamically
expected_input_shape = model.input_shape  # Example: (None, 64, 64, 3)
IMG_SIZE = expected_input_shape[1]  # Extract the correct size (e.g., 64)

print(f"Model expects input shape: {expected_input_shape}")

# Define the class labels corresponding to your 35 ISL classes
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '1', '2', '3', '4', '5', '6', '7', '8', '9']

def preprocess_image(image_path):
    """Loads an image, resizes it to the required size, normalizes, and expands dimensions."""
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Error loading image: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to model's expected input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Path to the test image (Change this to your actual image path)
image_path = r"C:\Users\Tanishka\Desktop\ISL\Indian\2\4.jpg"


try:
    img = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability

    # Map predicted class index to actual ISL sign
    predicted_label = class_labels[predicted_class]

    print(f"Prediction Scores: {prediction}")  # Show confidence scores for debugging
    print(f"Predicted Class Index: {predicted_class}")
    print(f"Predicted Sign: {predicted_label}")

except Exception as e:
    print(f"Error: {e}")
