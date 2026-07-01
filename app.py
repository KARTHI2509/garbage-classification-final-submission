import os
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

# Load classes
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load model path
model_path = os.getenv("MODEL_PATH", "MobileNetV3Large.keras")
if not os.path.exists(model_path):
    # Try same directory as script
    alt_path = os.path.join(os.path.dirname(__file__), model_path)
    if os.path.exists(alt_path):
        model_path = alt_path
    else:
        print(f"Warning: Model file not found at {model_path}. Please train your model or place your trained MobileNetV3Large.keras model file in this directory.")

model = None
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

def classify_image(img):
    if model is None:
        return "Error: Model file 'MobileNetV3Large.keras' not found. Please train the model first using train.py."
        
    # Resize image to 124x124 pixels (matching model training input shape)
    img = img.resize((124, 124))
    
    # Convert image to NumPy array with float32 type
    img_array = np.array(img, dtype=np.float32)
    
    # Expand dimensions to add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction (no manual preprocessing scaling because include_preprocessing=True is in the model)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    return f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})"

# Define Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Garbage Classification Web App",
    description="Upload an image of garbage (cardboard, glass, metal, paper, plastic, trash) to classify it."
)

if __name__ == "__main__":
    # Fetch deployment host and port dynamically
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    iface.launch(server_name=host, server_port=port)
