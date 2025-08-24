import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = ("best_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Cataract", "Conjunctivitis", "Eyelid_Drooping", "Normal", "Uveitis"]

IMG_SIZE = (224, 224)

def predict(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    confidences = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    predicted_class = class_names[np.argmax(preds)]
    return predicted_class, confidences


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Eye Image"),
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted Class"),
        gr.Label(num_top_classes=5, label="Class Probabilities")
    ],
    title="Eye Disease Detection",
    description="Upload an image of an eye and the model will predict possible conditions."
)



if __name__ == "__main__":
    demo.launch(share=True)