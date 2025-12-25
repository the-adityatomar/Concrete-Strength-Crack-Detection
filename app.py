import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from PIL import Image

# --- 1. CONFIGURATION & MODEL LOADING ---
print("--- Initializing AI Concrete Suite ---")

# Define paths based on your GitHub repo structure (Models/ folder)
# We use os.path.join to ensure it works on both Windows and Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Models')

# File names we defined in the notebooks
STRENGTH_MODEL_FILE = 'strength_model.keras'
CRACK_MODEL_FILE = 'crack_detection_model.keras'
SCALER_FILE = 'scaler.pkl'

# Helper function to find files (Checks 'Models/' folder first, then root)
def get_model_path(filename):
    path_in_folder = os.path.join(MODEL_DIR, filename)
    path_in_root = os.path.join(BASE_DIR, filename)
    
    if os.path.exists(path_in_folder):
        return path_in_folder
    elif os.path.exists(path_in_root):
        return path_in_root
    else:
        print(f"⚠️ Warning: {filename} not found in {MODEL_DIR} or root.")
        return None

# Load Scaler (Required for input transformation)
scaler_path = get_model_path(SCALER_FILE)
scaler = None
if scaler_path:
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")

# Load Strength Model (ANN)
strength_model_path = get_model_path(STRENGTH_MODEL_FILE)
strength_model = None
if strength_model_path:
    try:
        strength_model = tf.keras.models.load_model(strength_model_path)
        print(f"✅ Strength Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading strength model: {e}")

# Load Crack Detection Model (CNN)
crack_model_path = get_model_path(CRACK_MODEL_FILE)
crack_model = None
if crack_model_path:
    try:
        crack_model = tf.keras.models.load_model(crack_model_path)
        print(f"✅ Crack Detection Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading crack model: {e}")


# --- 2. LOGIC FUNCTIONS ---

def predict_strength_interface(cement, slag, fly_ash, water, superplast, coarse, fine, age):
    # Safety Check
    if strength_model is None or scaler is None:
        return "Error: System not fully loaded. Check server logs."

    # 1. Prepare the input exactly as the model expects (DataFrame)
    input_dict = {
        'cement': [cement],
        'slag': [slag],
        'fly_ash': [fly_ash],
        'water': [water],
        'superplasticizer': [superplast],
        'coarse_agg': [coarse],
        'fine_agg': [fine],
        'age': [age]
    }
    input_df = pd.DataFrame(input_dict)

    # 2. Scale (Using the loaded scaler)
    input_scaled = scaler.transform(input_df)

    # 3. Predict
    pred = strength_model.predict(input_scaled, verbose=0)[0][0]

    # 4. Return formatted string
    return f"{pred:.2f} MPa"


def detect_crack_interface(image):
    if image is None:
        return "Please upload an image."

    if crack_model is None:
        return "Error: Crack Detection Model not loaded."

    # 1. Preprocess the image to match training (128x128)
    # The model expects (Batch_Size, 128, 128, 3)
    image = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # 2. Predict
    prediction = crack_model.predict(img_array, verbose=0)
    score = prediction[0][0]

    # 3. Logic based on 'Positive' (Crack) vs 'Negative' (No Crack)
    if score > 0.5:
        confidence = score * 100
        return f"⚠️ CRACK DETECTED\nConfidence: {confidence:.2f}%"
    else:
        confidence = (1 - score) * 100
        return f"✅ Structure Healthy (No Crack)\nConfidence: {confidence:.2f}%"


# --- 3. UI LAYOUT ---
with gr.Blocks(title="AI Concrete Engineer") as demo:

    gr.Markdown("# 🏗️ AI Concrete Engineering Suite")
    gr.Markdown("### Design. Monitor. Maintain.")

    # TAB 1: Mix Design
    with gr.Tab("🧪 Mix Design Optimizer"):
        gr.Markdown("### Predict Compressive Strength based on Mix Proportions")

        with gr.Row():
            with gr.Column():
                cement = gr.Number(label="Cement (kg/m³)", value=350)
                slag = gr.Number(label="Blast Furnace Slag (kg/m³)", value=0)
                fly_ash = gr.Number(label="Fly Ash (kg/m³)", value=0)
                water = gr.Number(label="Water (kg/m³)", value=180)
            with gr.Column():
                superplast = gr.Number(label="Superplasticizer (kg/m³)", value=0)
                coarse = gr.Number(label="Coarse Aggregate (kg/m³)", value=1000)
                fine = gr.Number(label="Fine Aggregate (kg/m³)", value=750)
                age = gr.Number(label="Age (Days)", value=28)

        btn_predict = gr.Button("Predict Strength", variant="primary")
        out_strength = gr.Textbox(label="Predicted Compressive Strength", text_align="center", scale=2)

        # Connect button to function
        btn_predict.click(
            fn=predict_strength_interface,
            inputs=[cement, slag, fly_ash, water, superplast, coarse, fine, age],
            outputs=out_strength
        )

    # TAB 2: Inspection
    with gr.Tab("🔍 Structural Inspection"):
        gr.Markdown("### Upload Site Image for Automated Crack Detection")

        with gr.Row():
            in_image = gr.Image(type="pil", label="Site Photo")
            out_label = gr.Label(label="AI Assessment")

        btn_detect = gr.Button("Analyze Structure", variant="secondary")

        # Connect button to function
        btn_detect.click(
            fn=detect_crack_interface,
            inputs=in_image,
            outputs=out_label
        )

# Launch the app
print("Launching Dashboard...")
# Note: share=True generates a public link. Good for demo, bad for production security.
demo.launch(share=True)
