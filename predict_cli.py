import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Class definitions
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict(image_path, model_path="MobileNetV3Large.keras"):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train your model first using train.py.")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load and resize image to 124x124
    print(f"Processing image {image_path}...")
    img = Image.open(image_path).convert("RGB")
    img = img.resize((124, 124))
    
    # Convert image to NumPy array with float32 type
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction (no manual preprocessing scaling because include_preprocessing=True is in the model)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    print("\n" + "="*40)
    print(f"Prediction result:")
    print(f"  Class:      {predicted_class_name}")
    print(f"  Confidence: {confidence:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify garbage image using trained MobileNetV3Large model")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument("--model_path", type=str, default="MobileNetV3Large.keras", help="Path to the saved model file")
    args = parser.parse_args()

    predict(args.image_path, args.model_path)
