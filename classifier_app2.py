import streamlit as st
from transformers import CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer
import torch
import json
import os

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model(config_file, model_file):
    # Load model configuration
    config = CamembertConfig.from_pretrained(config_file)
    # Initialize model with the loaded configuration
    model = CamembertForSequenceClassification(config)
    # Load the state dictionary
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    model.eval()
    return model

# Function to evaluate a given text
def evaluate_text(model, tokenizer, input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt")
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the logits
    logits = outputs.logits
    # Get the predicted class index
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    return predicted_class_idx

# Streamlit app
st.title("French Language Level Evaluator")

st.sidebar.header("Upload Model Files")
config_file = st.sidebar.file_uploader("Upload config.json", type="json")
model_file = st.sidebar.file_uploader("Upload pytorch_model.bin", type="bin")

if config_file and model_file:
    # Save uploaded files
    with open("uploaded_config.json", "wb") as f:
        f.write(config_file.getbuffer())
    with open("uploaded_model.bin", "wb") as f:
        f.write(model_file.getbuffer())
    
    # Load model and tokenizer
    model = load_model("uploaded_config.json", "uploaded_model.bin")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    st.header("Enter French Sentence")
    input_text = st.text_area("French Sentence", "Bonjour, comment ça va?")

    if st.button("Evaluate"):
        # Evaluate the input text
        predicted_class_idx = evaluate_text(model, tokenizer, input_text)
        st.write(f"Predicted language level index: {predicted_class_idx}")

        # Optionally, you can map the index to actual labels if you have the mapping
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        st.write(f"Predicted language level: {levels[predicted_class_idx]}")

else:
    st.warning("Please upload both config.json and pytorch_model.bin files.")
