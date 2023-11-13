import warnings
import streamlit as st
import gc
from joblib import load
from PIL import Image
import re
import io
import os
import cv2
import xgboost
import pytesseract
from PIL import Image
from pix2tex.cli import LatexOCR
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import keras
import keras.backend as K
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import register_keras_serializable

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Define SelfAttention layer
@keras.utils.register_keras_serializable()
class SelfAttention(keras.layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros")
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.dot(x, self.W), axis=-1)
        et = et + self.b
        at = K.softmax(et, axis=1)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        output = K.sum(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def ocr(image):
    img = Image.open(image)
    model = LatexOCR()
    model(img)

    # Mention the installed location of Tesseract-OCR in your system
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    # Read image from which text needs to be extracted
    img = cv2.imread(image)

    # Preprocessing the image starts

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, 
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    # Creating a copy of the image
    im2 = img.copy()

    # A text file is created and flushed
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on the copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]

        # Open the file in append mode
        file = open("recognized.txt", "a")

        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)

        # Appending the text into the file
        file.write(text)
        file.write("\n")

        # Close the file
        file.close()

@st.cache_data
def read_data(file_name: str):
    
    physics =  (pd
                .read_csv(file_name)
                .drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Subject'])
                .rename(columns={'eng': 'question'}))
    
    return (physics
            .loc[(physics.topic == "Electric Fields") | (physics.topic == "Wave Motion") | 
                    (physics.topic == "Temperature and ideal Gasses") | (physics.topic == "Nuclear Physics") |
                    (physics.topic == "Forces") | (physics.topic == "D.C. Circuits") |
                    (physics.topic == "Gravitational Field") | (physics.topic == "Quantum Physics")]
            .assign(processed_question=lambda df_: df_['question'].apply(preprocess_text))
           )

# Function to clean and preprocess text
def preprocess_text(text):
    # Remove newline characters
    text = text.replace('\n', ' ')
    
    # Lowercase the text
    text = text.lower()
    
    # Remove numbers and punctuation
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Remove single-character words (like 'a', 'b', 'c')
    words = [word for word in words if len(word) > 1]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a string
    text = ' '.join(words)
    
    return text

@st.cache_resource
def load_vectorizer():
    primary_path = 'streamlit_gallery/utils/tfidf_vectorizer.joblib'
    alternative_path = '../../utils/tfidf_vectorizer.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Vectorizer not found in both primary and alternative directories!")

@st.cache_resource
def load_model_xgb():
    primary_path = 'streamlit_gallery/utils/best_model_physics_xgboost.joblib'
    alternative_path = '../../utils/best_model_physics_xgboost.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

@st.cache_resource 
def load_model_logreg():
    primary_path = 'streamlit_gallery/utils/best_model_physics_logreg.joblib'
    alternative_path = '../../utils/best_model_physics_logreg.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

@st.cache_resource        
def load_model_svc():
    primary_path = 'streamlit_gallery/utils/best_model_physics_svc.joblib'
    alternative_path = '../../utils/best_model_physics_svc.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        try:
            return load(alternative_path)
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")

@st.cache_resource
def load_model_lstm_attention():
    
    primary_path = 'streamlit_gallery/utils/best_model_physics_lstm_attention_legacy.h5'
    alternative_path = '../../utils/best_model_physics_lstm_attention_legacy.h5'
    
    try:
        # return load_model(primary_path, 
        #                   custom_objects={'SelfAttention': SelfAttention})
        # model_new = load(primary_path)
        # model_new.__class__.SelfAttention = SelfAttention
        
        model_new = load_model(primary_path)
        return model_new
    except FileNotFoundError:
        try:
            # return load_model(alternative_path, 
            #               custom_objects={'SelfAttention': SelfAttention})
            # model_new = load(alternative_path)
            # model_new.__class__.SelfAttention = SelfAttention
            
            model_new = load_model(alternative_path)
            return model_new
        except FileNotFoundError:
            raise Exception("Model not found in both primary and alternative directories!")
        
def preprocess_lstm(df, new_text):
    
    X = df['processed_question'].values
    y = df['topic'].values
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X)
    
    X = tokenizer.texts_to_sequences(X)
    maxlen = 100
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    
    new_text = tokenizer.texts_to_sequences([new_text])
    new_text = pad_sequences(new_text, padding='post', maxlen=maxlen)
    
    return new_text

# @st.cache_resource    
# def load_model_bert():
    
#     primary_path = 'streamlit_gallery/utils/best_model_physics_bert.h5'
#     alternative_path = '../../utils/best_model_physics_bert.h5'
    
#     loaded_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=8)
    
#     try:
#         loaded_model.load_state_dict(torch.load(primary_path, map_location=torch.device('cpu')), strict=False)
#         return loaded_model
#     except FileNotFoundError:
#         try:
#             loaded_model.load_state_dict(torch.load(alternative_path, map_location=torch.device('cpu')), strict=False)
#             return loaded_model
#         except FileNotFoundError:
#             raise Exception("Model not found in both primary and alternative directories!")

# def preprocess_bert(new_text):
#     # Tokenizer
#     tokenizerx = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#     user_input_encodings = tokenizerx(preprocess_text(new_text), truncation=True, padding=True, return_tensors='pt')
#     return user_input_encodings

def get_label_mapping():
    
    labelencoder = load('labelencoder_physics.joblib')
    return labelencoder

def make_prediction_hard_vote(loaded_models, 
                              input_data, 
                              label_mapping):
    
    predictions = []
    predictions.append(loaded_models[0].predict(input_data[0]))
    predictions.append(loaded_models[1].predict(input_data[0]))
    predictions.append(loaded_models[2].predict(input_data[0]))
    predictions.append([np.argmax(loaded_models[3].predict(input_data[1]))])
    # with torch.no_grad():
    #     output = loaded_models[4](**input_data[2])
    #     logits = output.logits
    #     predicted_class = torch.argmax(logits, dim=1).item()
    # predictions.append([predicted_class])
    
    final_prediction = stats.mode(predictions, axis=0, keepdims=True)[0]
    return label_mapping.inverse_transform(final_prediction[0])


def navigate_to_recommendation():
    st.experimental_set_query_params(p="recommendation")
    
def main():
    
    gc.enable()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    df = read_data("Subject/Physics/physics_labelled_updated.csv")
    
    # Title of the app
    st.subheader('Input Options')

    # Create tabs for image upload and text input
    tab1, tab2 = st.tabs(["🖼️ Image Upload", "✍️ Enter Text"])
    
    result = None
    uploaded_file = None
    user_input_text = None
    
    # Initialization
    if 'result' not in st.session_state:
        st.session_state['result'] = None
    
    with tab1:
        st.subheader("Image Upload")
        
        # File uploader allows user to add file
        uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            
            # To convert to a PIL Image object (if the file is an image)
            image = Image.open(io.BytesIO(bytes_data))
            
            file_extension = os.path.splitext(uploaded_file.name)[1]
            image_path = f"image{file_extension}"
            image.save(image_path)
    
            # Display the image
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            # Call the OCR function or whatever processing you need
            ocr('image.png')

            st.subheader('Extracted Text')
            # Display the recognized text
            result = open("recognized.txt", "r").read()
            st.write(result)
            st.session_state['result'] = result
            # uploaded_file = None
            
            hard_vote_prediction_result = make_prediction_hard_vote(loaded_models=[load_model_xgb(),
                                                                                   load_model_logreg(),
                                                                                   load_model_svc(),
                                                                                   load_model_lstm_attention(),
                                                                                #    load_model_bert(),
                                                                                ],
                                                                    input_data=[load_vectorizer().transform([preprocess_text(result)]),
                                                                                preprocess_lstm(df, preprocess_text(result)),
                                                                                # preprocess_bert(result)
                                                                                ],
                                                                    label_mapping=get_label_mapping())

            
            st.subheader(f"Predicted Topic: {hard_vote_prediction_result[0]}")
            st.session_state["predicted_topics"] = hard_vote_prediction_result[0]
            st.write("Need practice? Check out questions similar to this!")
            
            # Define your button and assign the navigation function to it
            practice_button = st.button("Practice!", key="practice_button1")

            if practice_button:
                # Navigate to the recommendation page
                navigate_to_recommendation()
                # Force a rerun of the script to reflect the query parameter change
                # st.rerun()
        
        else:
            st.warning("Please upload an image or enter text to get started!")

    with tab2:
        st.header("Text Input")
        
        user_input_text = st.text_area("Enter your text here...")
        
        if user_input_text != "":
            uploaded_file = None
            result = user_input_text
            st.subheader('Your Text')
            st.write(result)
            st.session_state['result'] = result
            
            hard_vote_prediction_result = make_prediction_hard_vote(loaded_models=[load_model_xgb(),
                                                                                   load_model_logreg(),
                                                                                   load_model_svc(),
                                                                                   load_model_lstm_attention(),
                                                                                #    load_model_bert(),
                                                                                ],
                                                                    input_data=[load_vectorizer().transform([preprocess_text(result)]),
                                                                                preprocess_lstm(df, preprocess_text(result)),
                                                                                # preprocess_bert(result)
                                                                                ],
                                                                    label_mapping=get_label_mapping())

            
            st.subheader(f"Predicted Topic: {hard_vote_prediction_result[0]}")
            st.session_state["predicted_topics"] = hard_vote_prediction_result[0]
            st.write("Need practice? Check out questions similar to this!")
            
            # Define your button and assign the navigation function to it
            practice_button = st.button("Practice!", key="practice_button2")

            if practice_button:
                # Navigate to the recommendation page
                navigate_to_recommendation()
                # Force a rerun of the script to reflect the query parameter change
                # st.rerun()
        
        else:
            st.warning("Please upload an image or enter text to get started!")
    
    gc.collect()

if __name__ == "__main__":
    main()