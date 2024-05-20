import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import os
import requests

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:  # Check if the request was successful
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write("Download completed successfully.")
    else:
        st.error(f"Failed to download file from URL: {url}")

def reassemble_file(file_path, num_parts):
    try:
        with open(file_path, 'wb') as output_file:
            for i in range(num_parts):
                part_file_path = f"{file_path}/pytorch_model.bin.part{i}"
                with open(part_file_path, 'rb') as part_file:
                    chunk = part_file.read()
                    output_file.write(chunk)
        return file_path
    except Exception as e:
        st.error(f"An error occurred while reassembling the file: {e}")
        return None

def evaluate_camembert_model(config_file, model_file, input_text):
    tokenizer = CamembertTokenizer.from_pretrained(config_file)
    model = CamembertForSequenceClassification.from_pretrained(config_file)
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    levels = ["A1","A2","B1","B2","C1","C2"]
    return levels[predicted_class]

# Streamlit interface
st.title("French Language Level Evaluator")

config_url = "https://raw.githubusercontent.com/juliw9/FrenchLanguageClassifier/main/config.json"
model_parts_urls = [
    f"https://raw.githubusercontent.com/juliw9/FrenchLanguageClassifier/main/model_pytorch/pytorch_model.bin.part{i}" 
    for i in range(6)
] 

input_text = st.text_input("Enter a French sentence:")

if st.button("Evaluate"):
    if input_text:
        os.makedirs("temp", exist_ok=True)
        config_file_path = os.path.join("temp", "config.json")
        download_file(config_url, config_file_path)

        model_file_path = os.path.join("temp", "model_pytorch")
        for i, url in enumerate(model_parts_urls):
            part_path = model_file_path + f"/pytorch_model.bin.part{i}"
            print("Downloading part", i, "from URL:", url)  # Print the URL being downloaded
            #download_file(url, part_path)
            print("Downloaded part", i)  # Print when each part is downloaded

        num_parts = len(model_parts_urls)
        print("Reassembling file...")
        #reassembled_file_path = reassemble_file(model_file_path, num_parts)
        
        if reassembled_file_path:
            print("File reassembled successfully")
        #    prediction = evaluate_camembert_model(config_file_path, reassembled_file_path, input_text)
        #    st.write(f"Predicted language level: {prediction}")
        else:
            st.error("Failed to reassemble the model file.")
    else:
        st.error("Please enter a sentence.")
