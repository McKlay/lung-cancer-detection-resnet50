import gradio as gr
import numpy as np
import tensorflow as tf
from utils import preprocess_image, generate_gradcam

# Load the model
model = tf.keras.models.load_model("model/resnet50_lung_model.h5")

# Prediction function
def predict_with_gradcam(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]
    label = "Cancer" if prediction > 0.5 else "No Cancer"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    
    heatmap_path = generate_gradcam(model, img_path)
    return f"{label} ({confidence*100:.2f}% confidence)", heatmap_path

# Gradio interface
interface = gr.Interface(
    fn=predict_with_gradcam,
    inputs=gr.Image(type="filepath", label="Upload CT Scan"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
    title="Lung Cancer Detection with Grad-CAM",
    description="Upload a CT scan. The model will predict Cancer / No Cancer and show a Grad-CAM heatmap of where it focused."
)

if __name__ == "__main__":
    interface.launch()
