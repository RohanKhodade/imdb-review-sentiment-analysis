import numpy as np
import tensorflow as  tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}


# loading pretrained model
model=load_model(r"C:\Users\HP\OneDrive\Desktop\Computer Vision\nlp project udemy\imdb.h5")


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


# function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2) +3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


### creating prediction function

def predict_sentiment(review):
    preprocessed=preprocess_text(review)
    
    prediction=model.predict(preprocessed)
    
    sentiment="positive" if prediction[0][0]>0.5 else "negative"
    
    return sentiment,prediction[0][0]

## designing streamlit app

import streamlit as st

st.title("Imdb movie review sentiment anaysis")

st.write("Enter a movie review to classify it as positive or negative")

user_input=st.text_area("movie review")

if st.button("classify"):
    
    preprocessed_input=preprocess_text(user_input)
    
    #make prediction
    prediction=model.predict(preprocessed_input)
    sentiment="positive" if prediction[0][0]>0.5 else "negative"
    
    st.write(f"Sentiment :{sentiment}")
    st.write(f"Prediction Score :{prediction[0][0]}")
    
else:
    st.write("enter a review")
