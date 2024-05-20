import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import os

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

def evaluate_camembert_model(config_file, model_file, input_text):
    # Load model configuration and tokenizer
    tokenizer = CamembertTokenizer.from_pretrained(config_file)
    model = CamembertForSequenceClassification.from_pretrained(config_file)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors='pt')
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    return predicted_class

# Streamlit interface
st.title("French Language Level Evaluator")

config_file = st.file_uploader("Upload the config.json file", type=["json"])
model_file_parts = st.file_uploader("Upload the split model files", type=["bin", "part"], accept_multiple_files=True)
input_text = st.text_input("Enter a French sentence:")

if st.button("Evaluate"):
    if config_file and model_file_parts and input_text:
        # Sort the uploaded parts by their suffix
        model_file_parts = sorted(model_file_parts, key=lambda x: x.name.split(".part")[-1])
        
        # Save uploaded config file
        config_file_path = os.path.join("temp", config_file.name)
        with open(config_file_path, "wb") as f:
            f.write(config_file.read())

        # Save uploaded parts
        os.makedirs("temp", exist_ok=True)
        for part in model_file_parts:
            part_path = os.path.join("temp", part.name)
            with open(part_path, "wb") as f:
                f.write(part.read())
        
        # Reassemble the model file
        model_file_path = os.path.join("temp", "pytorch_model.bin")
        num_parts = len(model_file_parts)
        reassembled_file_path = reassemble_file(model_file_path, num_parts)
        
        if reassembled_file_path:
            # Evaluate the input text
            prediction = evaluate_camembert_model(config_file_path, reassembled_file_path, input_text)
            st.write(f"Predicted language level: {prediction}")
        else:
            st.error("Failed to reassemble the model file.")
    else:
        st.error("Please upload the config file, model files, and enter a sentence.")
