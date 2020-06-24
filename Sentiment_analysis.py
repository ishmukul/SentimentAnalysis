"""
Scrip file for doing sentiment analysis.
A Recurrent Neural Network model is loaded from saved model file 'models' folder.
Model is trained with 1.6M tweets.
MaxLen of sequence for analysis is 24.

predict function predicts the sentiment of a single line text.

Summary of accuracies of model:
    Accuracy of NeuralNet on validation set is 81.62
    Accuracy of NeuralNet on test set is 80.92

"""

# Import libraries

import os  # Some OS operations
from time import perf_counter as timer
import pandas as pd  # Manipulating data

# Keras libraries for Neural Networks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import h5py
import pickle
from utilities import *

# ===============================
# Load tokenizer and model
t_start = timer()  # Clock start

MaxLen = 24  # Do not change because model already trained with these parameters

# Load tokenizer
with open('models/tokenizer_10k.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load Keras model
model_name = "rnn_glove_model"
model_name = "rnn_emb_split01_model"
model_name = "rnn_emb_model"
# model_name = "nn_w2v_model"
# model_name = "emb_model"

model = load_model("models/" + model_name + ".h5")

print(model.summary())
# ===================================
# Function for predicting sentiment of sentence
def predict(input_text, thresh=0.5, max_length=24):
    text = text_clean(input_text)
    test_seq = tokenizer.texts_to_sequences(text)
    test = pad_sequences(test_seq, maxlen=max_length)

    s = model.predict(test)
    if s > thresh:
        print("Positive")
    else:
        print("Negative")
    return s


# ===============================
# Test some cases
print(predict("You have written a fantastic review about movie but things have changed."))
print(predict("I have to say your review is not good."))
print(predict("I have to say your review is nt good."))
print(predict("I have to say your review is not bad."))

# In case, interested in model summary
# print(rnn_emb_model.summary())

# ===================================
# Output in terminal
# Positive
# [[0.8607207]]
# Negative
# [[0.1286704]]
# Negative
# [[0.36957595]]
# Positive
# [[0.75851214]]

# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 24, 50)            500000
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, 24, 128)           58880
# _________________________________________________________________
# bidirectional_2 (Bidirection (None, 64)                41216
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                4160
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 64)                0
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 65
# =================================================================
# Total params: 604,321
# Trainable params: 604,321
# Non-trainable params: 0
# _________________________________________________________________
# None
