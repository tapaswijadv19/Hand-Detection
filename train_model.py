import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

#  Set dataset path
DATASET_PATH = "dataset"

# Image parameters
IMG_SIZE = 64  # Resize images to 64x64
BATCH_SIZE = 32  # Adjust based on GPU/CPU memory

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # 80% Train, 20% Validation
)

#  Load dataset
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#  Get number of classes
NUM_CLASSES = len(train_data.class_indices)

#  Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅Train Model
EPOCHS = 40
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

#  Evaluate Model
val_loss, val_acc = model.evaluate(val_data)
print(f" Validation Accuracy: {val_acc * 100:.2f}%")

#  Save Model
model.save("isl_detection_model.keras")

#  Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#  Save the TFLite Model
with open("isl_model.tflite", "wb") as f:
    f.write(tflite_model)

print(" Model training complete & saved as 'isl_detection_model.keras' & 'isl_model.tflite'")

