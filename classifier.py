import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import os
import requests
import base64

# Helper function to convert image to base64 from a URL
def get_image_base64_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode()
    else:
        st.error(f"Failed to download image from URL: {image_url}")
        return None

# Function to evaluate the Camembert model
def evaluate_camembert_model(model_name, input_text):
    model = CamembertForSequenceClassification.from_pretrained(model_name)
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model.eval()
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
        .source-text {{
            font-size: 0.8em;
            color: black;
            margin-top: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Dictionary mapping levels to descriptions
level_descriptions = {
    "A1": "Your friend can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type. They introduce themselves and others and can ask and answer questions about personal details such as where they live, people they knows and things they have. They can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.",
    "A2": "Your friend can understand sentences and frequently used expressions related to areas of most immediate relevance (e.g. very basic personal and family information, shopping, local geography, employment). They can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters. They can describe in simple terms aspects of his/her background, immediate environment and matters in areas of immediate need.",
    "B1": "Your friend can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc. They can deal with most situations likely to arise whilst travelling in an area where the language is spoken. They produce simple connected text on topics which are familiar or of personal interest. Can describe experiences and events, dreams, hopes & ambitions and briefly give reasons and explanations for opinions and plans.",
    "B2": "Your friend can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in their field of specialisation. They can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party. They produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.",
    "C1": "Your friend can understand a wide range of demanding, longer texts, and recognise implicit meaning. They express themselves fluently and spontaneously without much obvious searching for expressions. They can use language flexibly and effectively for social, academic and professional purposes. They can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.",
    "C2": "Your friend can understand with ease virtually everything heard or read. They can summarise information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation. They can express themselves spontaneously, very fluently and precisely, differentiating finer shades of meaning even in more complex situations."
}

# Streamlit interface
st.title("French Language Level Evaluator")

# Hugging Face model name
model_name = "juliw9/FrenchModel"

# Get input text from user
input_text = st.text_input("Enter a French sentence:")

# Set background image
image_url = "https://raw.githubusercontent.com/juliw9/FrenchLanguageClassifier/main/StreamlitApp/Drapeau_de_la_France.png"
image_base64 = get_image_base64_from_url(image_url)
if image_base64:
    set_background(image_base64)

if st.button("Evaluate"):
    if input_text:
        # Evaluate the model with the input text
        predicted_class_idx = evaluate_camembert_model(model_name, input_text)
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        predicted_level = levels[predicted_class_idx]
        st.markdown(f'<div class="result-text">Predicted language level: {predicted_level}</div>', unsafe_allow_html=True)
        description = level_descriptions[predicted_level]
        st.markdown(f'<div class="description-text">{description}</div>', unsafe_allow_html=True)
        # Add source line
        st.markdown(
            f'<div class="source-text">Source: <a href="https://www.coe.int/en/web/common-european-framework-reference-languages/table-1-cefr-3.3-common-reference-levels-global-scale" target="_blank">Common European Framework of Reference for Languages</a></div>', 
            unsafe_allow_html=True
        )
    else:
        st.error("Please enter a sentence.")
