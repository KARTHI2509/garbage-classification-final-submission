from pathlib import Path

import gradio as gr
import numpy as np
import json
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
model = load_model(BASE_DIR / "garbage_classifier.keras")

with open(BASE_DIR / "class_names.json", "r") as f:
    class_names = json.load(f)

def classify_image(img):
    img = img.resize((160,160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    top_3_idx = predictions.argsort()[-3:][::-1]

    result = {}
    for idx in top_3_idx:
        result[class_names[idx].capitalize()] = float(predictions[idx])

    return result


custom_css = """
body {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #334155, #0f172a);
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# ♻ Smart Garbage Classification System")

    with gr.Row():
        image_input = gr.Image(type="pil")
        output_label = gr.Label(num_top_classes=3)

    btn = gr.Button("Analyze")

    btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=output_label
    )


if __name__ == "__main__":
    demo.launch()