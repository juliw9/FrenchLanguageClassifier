import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification, CamembertConfig
import torch
import os
import requests

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        #st.write(f"Downloaded: {output_path}")
    else:
        st.error(f"Failed to download file from URL: {url}")

# Function to reassemble the model file from parts
def reassemble_file(file_path, num_parts):
    try:
        with open(file_path, 'wb') as output_file:
            for i in range(num_parts):
                part_file_path = f"{file_path}.part{i}"
                with open(part_file_path, 'rb') as part_file:
                    chunk = part_file.read()
                    output_file.write(chunk)
        return file_path
    except Exception as e:
        st.error(f"An error occurred while reassembling the file: {e}")
        return None

# Function to evaluate the Camembert model
def evaluate_camembert_model(config_file, model_file, input_text):
    config = CamembertConfig.from_pretrained(config_file)
    model = CamembertForSequenceClassification(config)
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    return predicted_class_idx

# CSS for background image and white rectangle
def set_background(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{image_path}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .text-container {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px;
            z-index: 1;
            position: relative;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
# Streamlit interface
st.title("French Language Level Evaluator")

# URLs for the config and model files
config_url = "https://raw.githubusercontent.com/juliw9/FrenchLanguageClassifier/main/config.json"
model_parts_urls = [
    f"https://raw.githubusercontent.com/juliw9/FrenchLanguageClassifier/main/pytorch_model.bin.part{i}" 
    for i in range(6)
]

# Create temporary directory for downloading files
os.makedirs("temp", exist_ok=True)

# Get input text from user
input_text = st.text_input("Enter a French sentence:")

# Set background image
image_path = get_image_base64("Drapeau_de_la_France.png")
set_background(image_path)

if st.button("Evaluate"):
    if input_text:
        # Download the config file
        config_file_path = os.path.join("temp", "config.json")
        download_file(config_url, config_file_path)

        # Download and reassemble the model file
        model_file_path = os.path.join("temp", "pytorch_model.bin")
        for i, url in enumerate(model_parts_urls):
            part_path = model_file_path + f".part{i}"
            #st.write("Downloading part", i, "from URL:", url)
            download_file(url, part_path)
            #st.write("Downloaded part", i)

        num_parts = len(model_parts_urls)
        #st.write("Reassembling file...")
        reassembled_file_path = reassemble_file(model_file_path, num_parts)

        if reassembled_file_path:
            #st.write("File reassembled successfully")
            # Evaluate the model with the input text
            predicted_class_idx = evaluate_camembert_model(config_file_path, reassembled_file_path, input_text)
            levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
            predicted_level = levels[predicted_class_idx]
            st.write(f"Predicted language level: {predicted_level}")
        else:
            st.error("Failed to reassemble the model file.")
    else:
        st.error("Please enter a sentence.")
