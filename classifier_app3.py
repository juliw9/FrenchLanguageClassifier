import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification, CamembertConfig
import torch
import os
import requests
import base64

# Helper function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to download files
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
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
        .result-text {{
            color: black;
            font-size: 2em;
            font-weight: bold;
        }}
        .description-text {{
            font-size: 1em;
            color: black;
            margin-top: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Dictionary mapping levels to descriptions
level_descriptions = {
    "A1": "Can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type. Can introduce him/herself and others and can ask and answer questions about personal details such as where he/she lives, people he/she knows and things he/she has. Can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.",
    "A2": "Can understand sentences and frequently used expressions related to areas of most immediate relevance (e.g. very basic personal and family information, shopping, local geography, employment). Can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters. Can describe in simple terms aspects of his/her background, immediate environment, and matters in areas of immediate need.",
    "B1": "Can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc. Can deal with most situations likely to arise whilst travelling in an area where the language is spoken. Can produce simple connected text on topics that are familiar or of personal interest. Can describe experiences and events, dreams, hopes, and ambitions and briefly give reasons and explanations for opinions and plans.",
    "B2": "Can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialization. Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party. Can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.",
    "C1": "Can understand a wide range of demanding, longer texts, and recognize implicit meaning. Can express him/herself fluently and spontaneously without much obvious searching for expressions. Can use language flexibly and effectively for social, academic, and professional purposes. Can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organizational patterns, connectors, and cohesive devices.",
    "C2": "Can understand with ease virtually everything heard or read. Can summarize information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation. Can express him/herself spontaneously, very fluently, and precisely, differentiating finer shades of meaning even in more complex situations."
}

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
            download_file(url, part_path)

        num_parts = len(model_parts_urls)
        reassembled_file_path = reassemble_file(model_file_path, num_parts)

        if reassembled_file_path:
            # Evaluate the model with the input text
            predicted_class_idx = evaluate_camembert_model(config_file_path, reassembled_file_path, input_text)
            levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
            predicted_level = levels[predicted_class_idx]
            st.markdown(f'<div class="result-text">Predicted language level: {predicted_level}</div>', unsafe_allow_html=True)
            description = level_descriptions[predicted_level]
            st.markdown(f'<div class="description-text">{description}</div>', unsafe_allow_html=True)
        else:
            st.error("Failed to reassemble the model file.")
    else:
        st.error("Please enter a sentence.")
