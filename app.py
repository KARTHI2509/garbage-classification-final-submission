import gradio as gr
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("garbage_classifier.keras")

# Class names (must match training order)
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Waste disposal tips
waste_tips = {
    "cardboard": "♻ Recycle in dry waste bin.",
    "glass": "⚠ Handle carefully and recycle separately.",
    "metal": "🔩 Recycle at scrap collection centers.",
    "paper": "📄 Reuse or recycle in paper waste bin.",
    "plastic": "🧴 Separate by type before recycling.",
    "trash": "🗑 Dispose in general waste."
}


def classify_image(image):
    try:
        # Resize image
        image = image.resize((124, 124))
        image = np.array(image)

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Handle RGBA images
        if len(image.shape) == 3 and image.shape[-1] == 4:
            image = image[:, :, :3]

        # Expand dimensions
        image = np.expand_dims(image, axis=0)

        # Predict
        predictions = model.predict(image, verbose=0)[0]

        # Top 3 predictions
        top3_idx = np.argsort(predictions)[-3:][::-1]

        top_predictions = {
            class_names[i]: float(predictions[i])
            for i in top3_idx
        }

        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100

        tip = waste_tips[predicted_class]

        result = f"""
## ♻ Classification Result

### 🏷 Category: **{predicted_class.upper()}**

### 🎯 Confidence: **{confidence:.2f}%**

### 💡 Disposal Tip:
{tip}
"""

        return result, top_predictions

    except Exception as e:
        return f"❌ Error: {str(e)}", {}


# Custom theme
custom_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="blue",
    neutral_hue="gray"
)

# Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="📤 Upload Waste Image"),
    outputs=[
        gr.Markdown(label="📋 Prediction Result"),
        gr.Label(num_top_classes=3, label="📊 Top Predictions")
    ],
    title="♻ Smart Garbage Classification System",
    description="""
Upload an image of waste material and let AI classify it instantly.

### Supported Waste Types:
📦 Cardboard  
🍾 Glass  
🔩 Metal  
📄 Paper  
🧴 Plastic  
🗑 Trash
"""
)

# Launch
iface.launch(theme=custom_theme)