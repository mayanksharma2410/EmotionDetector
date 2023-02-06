# Importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import streamlit as st
import numpy as np
import re
from PIL import Image

# Label categories
class_cat = {0 : 'Sadness', 1 : 'Joy', 2 : 'Love', 3 : 'Anger', 4 : 'Fear', 5 : 'Surprise'}

# Loading the model
model = load_model('model.h5')

# Loading the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Function to remove special characters and links
def remove_special_characters(text):
    # Regular expression pattern for URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Replace URLs with an empty string
    text = re.sub(url_pattern, '', text)

    # Regular expression pattern for mentions starting with @
    mention_pattern = re.compile(r'@\w+')

    # Replace mentions with an empty string
    text = re.sub(mention_pattern, '', text)

    # Regular expression pattern for special characters
    special_char_pattern = re.compile(r'[^\w\s]+')

    # Replace special characters with an empty string
    text = re.sub(special_char_pattern, '', text)

    return text

# Function to pre-process the data
def pre_process(data, tokenizer):
    max_sequence_len = 66
    # Calling function to remove special characters
    data = remove_special_characters(data).lower()
    
    # Tokenizing the text
    data_seq = tokenizer.texts_to_sequences([data])

    # Padding the sequence
    data_pad = np.array(pad_sequences(data_seq, maxlen=max_sequence_len, padding='post'))

    return data_pad

st.title("Emotion Detector")
# Sentence# sentence = "@mayank I WILL try and stay focused in order to avoid that feeling of a reluctant finish! https://www.google.com"

sentence = st.text_input("Enter the text to predict its emotion", help="Please enter the text or tweet of which you want the emotion prediction.")

if st.button('Predict'):
    # Calling pre-processing function
    processed_data = pre_process(sentence, tokenizer)

    # Predicting the emotions
    prediction = model.predict(np.expand_dims(processed_data[0], axis=0))[0]
    pred_class = class_cat[np.argmax(prediction).astype('uint8')]
    st.write('**Actual Sentence:**', sentence) 
    # st.write(sentence)
    st.write(np.argmax(prediction).astype('uint8'))
    emoji = Image.open('Emoji/' + pred_class + '.png')

    col1, col2 = st.columns([1,3])

    with col1:
        st.write("**Emotion Predicted:**")
        st.write(pred_class)

    with col2:
        st.image(emoji, width=50)

else:
    print('')