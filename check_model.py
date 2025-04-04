import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("isl_detection_model.keras")

# Print model summary
model.summary()
