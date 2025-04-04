import os

dataset_path = r"C:\Users\Tanishka\Desktop\ISL\Indian"
class_counts = {}

for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_folder):
        class_counts[class_name] = len(os.listdir(class_folder))

print("Dataset Class Distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")
