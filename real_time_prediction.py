import cv2
import numpy as np
import tensorflow as tf
import os
import time

# ✅ Load trained model
model = tf.keras.models.load_model("isl_detection_model.keras")

# ✅ Class Labels (1-9, A-Z)
labels = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
label_dict = {i: labels[i] for i in range(len(labels))}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (64, 64))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dims for model
    return img.astype(np.float32)

def predict_image():
    """Predict ISL gesture from an image"""
    print("\nEnter the file path in raw string format like this:")
    print('Example: r"C:\\Users\\Tanishka\\Desktop\\ISL\\A\\17.jpg"')

    file_path = input("\nEnter image file path: ").strip()
    if file_path.startswith('r"') and file_path.endswith('"'):
        file_path = file_path[2:-1]

    if not os.path.exists(file_path):
        print("❌ Error: File not found. Please check the path and try again.")
        return

    image = cv2.imread(file_path)
    if image is None:
        print("❌ Error: Unable to read image. Please check the file format.")
        return

    img = preprocess_image(image)
    predictions = model.predict(img)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    print(f"✅ Predicted Gesture: {label_dict[predicted_class]} (Confidence: {confidence:.2f})")

    cv2.putText(image, f"{label_dict[predicted_class]} ({confidence:.2f})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Predicted Image", image)
    cv2.waitKey(3000)  # Show image for 3 seconds
    cv2.destroyAllWindows()

def predict_video():
    """Predict ISL gestures from a video file"""
    print("\nEnter the file path in raw string format like this:")
    print('Example: r"C:\\Users\\Tanishka\\Desktop\\ISL\\video.mp4"')

    file_path = input("\nEnter video file path: ").strip()
    if file_path.startswith('r"') and file_path.endswith('"'):
        file_path = file_path[2:-1]

    if not os.path.exists(file_path):
        print("❌ Error: File not found. Please check the path and try again.")
        return

    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess_image(frame)
        predictions = model.predict(img)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        text = f"Detected: {label_dict[predicted_class]} ({confidence:.2f})"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ISL Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def predict_webcam():
    """Predict ISL gestures in real-time using a webcam"""
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess_image(frame)
        predictions = model.predict(img)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        # ✅ Draw text & FPS
        text = f"{label_dict[predicted_class]} ({confidence:.2f})"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ✅ FPS Calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("ISL Detection - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('p'):  # Press 'p' to pause
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# ✅ Main loop for user input selection
while True:
    print("\n🔹 Select input type:")
    print("1️⃣ Image")
    print("2️⃣ Video")
    print("3️⃣ Webcam")
    print("4️⃣ Exit")

    option = input("Enter option (1/2/3/4): ").strip()

    if option == '1':
        predict_image()
    elif option == '2':
        predict_video()
    elif option == '3':
        predict_webcam()
    elif option == '4':
        print("🚀 Exiting program.")
        break
    else:
        print("❌ Invalid option. Please enter 1, 2, 3, or 4.")
