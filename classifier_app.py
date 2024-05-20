import streamlit as st
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch
import json
import os

def reassemble_file(chunk_dir, output_path):
    chunk_number = 0
    with open(output_path, 'wb') as output_file:
        while True:
            chunk_file_name = os.path.join(chunk_dir, f"model.safetensors.part{chunk_number}")
            if not os.path.exists(chunk_file_name):
                break
            with open(chunk_file_name, 'rb') as chunk_file:
                output_file.write(chunk_file.read())
            chunk_number += 1

# Directory where the chunks are stored
model_chunks_dir = "model"
model_reassembled_path = "reassembled_model.safetensors"

if not os.path.exists(model_reassembled_path):
    st.write("Reassembling model file...")
    reassemble_file(model_chunks_dir, model_reassembled_path)
    st.write("Model file reassembled successfully!")

# Load the configuration
config_path = "config.json"

# Load the tokenizer
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Load the configuration from the file
with open(config_path, 'r') as f:
    config = json.load(f)

# Load the model
model = CamembertForSequenceClassification.from_pretrained(
    'camembert-base', state_dict=torch.load(model_reassembled_path), config=config
)

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
