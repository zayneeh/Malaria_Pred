import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your saved model (.keras file)
model = load_model("malaria_model.keras")

IMG_SIZE = (64, 64)   # matches your training

def predict(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalization

    prediction = model.predict(img_array)[0][0]
    label = "Infected" if prediction > 0.5 else "Uninfected"
    return label

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Prediction"),
    title="Malaria Cell Image Predictor",
    description="Upload a blood smear cell image to detect malaria (Infected/Uninfected)."
).launch()
