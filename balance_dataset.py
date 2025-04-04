import os
import cv2
import numpy as np
import random

# ✅ Define dataset path and target count
dataset_path = r"C:\Users\Tanishka\Desktop\ISL\Indian"
target_count = 1200  # Set target images per class

# ✅ Define augmentation functions
def random_rotation(image):
    """Apply random rotation to image"""
    angle = random.randint(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def random_flip(image):
    """Apply random horizontal flip"""
    return cv2.flip(image, 1) if random.random() > 0.5 else image

def adjust_brightness(image):
    """Apply random brightness adjustment"""
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_image(image):
    """Apply a combination of augmentations"""
    image = random_rotation(image)
    image = random_flip(image)
    image = adjust_brightness(image)
    return image

# ✅ Loop through each class and balance images
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue

    images = os.listdir(class_folder)
    current_count = len(images)

    # ✅ Delete extra images if more than 1200
    if current_count > target_count:
        print(f"Deleting extra images in {class_name} ({current_count} → {target_count})...")
        extra_images = images[target_count:]  # Get extra images
        for img_name in extra_images:
            os.remove(os.path.join(class_folder, img_name))

    # ✅ Augment images if less than 1200
    elif current_count < target_count:
        print(f"Augmenting {class_name} ({current_count} → {target_count})...")
        while len(os.listdir(class_folder)) < target_count:
            img_name = random.choice(images)
            img_path = os.path.join(class_folder, img_name)

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue  # Skip if image not found

            # Apply augmentation
            augmented_img = augment_image(image)

            # Save augmented image
            new_img_name = f"aug_{len(os.listdir(class_folder))}.jpg"
            new_img_path = os.path.join(class_folder, new_img_name)
            cv2.imwrite(new_img_path, augmented_img)

print("✅ Dataset balancing complete! All classes now have exactly 1200 images.")
