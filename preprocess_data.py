import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define constants
IMG_SIZE = 64  # Resize images to 64x64
DATASET_PATH = "dataset"
classes = sorted(os.listdir(DATASET_PATH))  # Extract class labels
num_classes = len(classes)

# Data and labels
data, labels = [], []

# Load images
for label, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_PATH, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize to 0-1
        data.append(img)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

# Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)
