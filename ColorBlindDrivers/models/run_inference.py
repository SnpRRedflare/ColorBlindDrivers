import cv2
import numpy as np
import tensorflow as tf
import pyttsx3


# Load model
model = tf.keras.models.load_model("../models/traffic_light_model.h5")
class_mapping = {0: "Red Light", 1: "Yellow Light", 2: "Green Light"}


# Initialize text-to-speech
engine = pyttsx3.init()


def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    label = class_mapping[class_id]


    # Audio feedback
    engine.say(f"{label} detected")
    engine.runAndWait()
    return label


# Example usage
result = classify_image("../data/sample_image.jpg")
print(f"Detected: {result}")