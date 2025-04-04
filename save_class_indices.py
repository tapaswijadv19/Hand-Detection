import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset path (same as used during training)
DATASET_PATH = "dataset"

# Image parameters
IMG_SIZE = 64
BATCH_SIZE = 32

# Data Augmentation (Only for loading dataset, no training)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load dataset (Only to get class indices)
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Save class indices to a JSON file
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Class indices saved to 'class_indices.json' successfully!")
