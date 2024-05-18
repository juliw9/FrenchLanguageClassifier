import streamlit as st
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch
import json
import requests
import os

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = requests.get(url, params=params, stream=True)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

# File paths
config_path = "config.json"
model_path = "model.safetensors"
file_id = "1HbAX8pMeSNxv_ZTSItoOIV0GWcWIatNI"  # Your file ID from Google Drive

# Check if model file exists, if not download it
if not os.path.exists(model_path):
    st.write("Downloading model file...")
    download_file_from_google_drive(file_id, model_path)
    st.write("Model file downloaded successfully!")

# Load the configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# Load the tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertForSequenceClassification.from_pretrained('camembert-base', state_dict=torch.load(model_path), config=config)

# Define a function to predict difficulty
def predict_difficulty(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return config['id2label'][str(predicted_class)]

# Streamlit app
st.title("French Sentence Difficulty Classifier")

sentence = st.text_input("Enter a French sentence:")

if sentence:
    difficulty = predict_difficulty(sentence)
    st.write(f"The predicted difficulty level is: {difficulty}")
