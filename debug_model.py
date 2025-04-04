import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("isl_detection_model.keras")  # Replace with your actual model file

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the class_indices dictionary
class_labels = {v: k for k, v in class_indices.items()}

# Load test image
img_path = r"C:\Users\Tanishka\Desktop\ISL\Indian\6\0.jpg" # Change this to your actual test image path
img = image.load_img(img_path, target_size=(64, 64))  # Resize to match training data
img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Show the image
plt.imshow(img)
plt.axis("off")
plt.title(r"C:\Users\Tanishka\Desktop\ISL\Indian\6\0.jpg")
plt.show()

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Get predicted label
predicted_label = class_labels[predicted_class]
print(f"Predicted class: {predicted_label}")
