import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# Paths
DATASET_PATH = "./dataset"  # Update to your dataset's base directory
CLASS_MAPPING = {"red": 0, "yellow": 1, "green": 2}  # Update as needed


def load_and_preprocess_data():
    images, labels = [], []


    # Loop through each folder in the dataset directory
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)


        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)


                # Read and preprocess the image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                images.append(img)


                # Assign a label based on the folder name (or other logic)
                label = CLASS_MAPPING.get(folder_name, -1)
                if label == -1:
                    continue
                labels.append(label)


    # Normalize and split data
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)


    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


# Example usage
X_train, X_val, y_train, y_val = load_and_preprocess_data()
print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
