import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras.models import Sequential
from keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint




# Paths
DATASET_PATH = "./data"
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_PATH, "annotations.json")




# Class mapping
CLASS_MAPPING = {"red": 0, "yellow": 1, "green": 2}




def load_and_preprocess_data():
    with open(ANNOTATIONS_FILE, 'r') as f:
        annotations = json.load(f)




    images, labels = [], []




    for item in annotations:
        img_path = os.path.join(IMAGES_DIR, item['path'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to match the model input size
        images.append(img)




        # Extract the class label
        label = CLASS_MAPPING[item['annotations'][0]['class']]
        labels.append(label)




    # Normalize and split data
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)




    return train_test_split(images, labels, test_size=0.2, random_state=42)




def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes: red, yellow, green
    ])
    return model




def main():
    # Load data
    X_train, X_val, y_train, y_val = load_and_preprocess_data()




    # Create model
    model = create_model()




    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])




    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint('traffic_light_model.h5', save_best_only=True)
    ]




    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks
    )




    # Save the final model
    model.save("final_traffic_light_model.h5")
    print("Model training completed and saved.")




if __name__ == "__main__":
    main()