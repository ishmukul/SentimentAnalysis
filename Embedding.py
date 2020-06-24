"""
A model using Word embedding training for sentiment analysis.
Embedding layer is trained on full set of 1.6M tweets and 10k word vocabulary.

Accuracies are pretty good.
Training ~ 83%
Validation ~ 79%
Test ~ 76%
"""

# Import libraries

import os  # Some OS operations
from time import perf_counter as timer
import string
import numpy as np
import pandas as pd  # Manipulating data
from sklearn.utils import shuffle  # Shuffle data, in case we need sample of data
import matplotlib.pyplot as plt
from collections import Counter
import heapq
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Keras libraries for Neural Networks
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Adadelta, RMSprop

from utilities import *

# ===============================================
# Load files
data_training = pd.read_csv('data/training/training_tweets_clean.csv')
data_training = data_training.replace(np.nan, '', regex=True)  # Some values were blank. Pandas read them as nan.

data_validation = pd.read_csv('data/validation/validation_tweets_clean.csv')
# data_validation = data_validation.replace(np.nan, '', regex=True)  # Some values were blank. Pandas read them as nan.

data_test = pd.read_csv("data/test/test_tweets_clean.csv")
data_test = data_test.replace(np.nan, '', regex=True)  # Some values were blank. Pandas read them as nan.

# Further slicing data for selective tweets.  Shuffle data if taking a slice of data.
# It is important because data contains tweets sorting based on Polarity
# data_training = shuffle(data_training)  # Shuffle the data
# data_training.reset_index(inplace=True, drop=True)
# data_training = data_training[:2000]

print("All files loaded")
print("Working with %d tweets", len(data_training))

# Drop indices for neutral reviews
to_drop = [2]
data_validation = data_validation[~data_validation['Polarity'].isin(to_drop)]
data_test = data_test[~data_test['airline_sentiment'].isin(to_drop)]

# Tweets corresponding to training, validation and test set
tweets_training = data_training['Tweet']
tweets_validation = data_validation['Tweet']
tweets_test = data_test['text']

# ===============================
# Preparing for inputs for embedding model
t_start = timer()
VocLen = 10000  # Number of words to keep in dictionary in dictionary
tk = Tokenizer(num_words=VocLen)
tk.fit_on_texts(tweets_training)

# Converting sequences of tweets
X_train_seq = tk.texts_to_sequences(tweets_training)
X_valid_seq = tk.texts_to_sequences(tweets_validation)
X_test_seq = tk.texts_to_sequences(tweets_test)

# Check sequence lengths of all tweets, i.e. length of Tweets' words.
seq_lengths = tweets_training.apply(lambda x: len(x.split(' ')))
# print(seq_lengths.value_counts())

# Every run is random due to shuffling of dataset and selecting 100K tweets. In this run, we have only 3 tweets more
# than 24 length. Ignoringing them and truncating tweets at 24 max length.
# Out of 1.6M tweets nearly 40 reduced tweets have more than 24 length.
MaxLen = 24
X_train = pad_sequences(X_train_seq, maxlen=MaxLen)
X_valid = pad_sequences(X_valid_seq, maxlen=MaxLen)
X_test = pad_sequences(X_test_seq, maxlen=MaxLen)
# print(X_train_seq_trunc[10])  # Example of padded sequence

y_train = data_training['Polarity']
y_valid = data_validation['Polarity']
y_test = data_test['airline_sentiment']
t_stop = timer()
print("Time to pre-process model of %d words vocabulary for %d Tweets is %0.5f s" % (
    VocLen, len(data_training), (t_stop - t_start)))

# =================================
# Create Embedding model
t_start = timer()


def define_model():
    modl = Sequential()
    modl.add(Embedding(VocLen, 50, input_length=MaxLen))
    modl.add(Flatten())
    modl.add(Dense(1, activation='sigmoid'))
    return modl


emb_model = define_model()
print(emb_model.summary())
# Optimizer algorithms
# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # Stochastic Gradient Descent
# opt = RMSprop(lr=0.001, rho=0.9)
# opt = Adagrad(learning_rate=0.01)
opt = Adadelta(learning_rate=1.0, rho=0.95)
# opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Loss functions, several options available
# loss = 'mean_squared_error'
# loss = 'mean_absolute_error'
loss = 'binary_crossentropy'
# loss = 'categorical_crossentropy'

emb_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

file_path = "models/best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor="val_loss",
                              verbose=1, save_best_only=True, mode="min")
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

emb_history = emb_model.fit(X_train, y_train, epochs=50, batch_size=512, validation_data=[X_valid, y_valid],
                            verbose=2, callbacks=[check_point, early_stop])
# emb_history = emb_model.fit(X_train, y_train, epochs=20, batch_size=512, validation_split=0.01, verbose=2)
print(emb_history.history['accuracy'][-1])

score_valid = emb_model.evaluate(X_valid, y_valid, batch_size=512, verbose=2)
score_test = emb_model.evaluate(X_test, y_test, batch_size=512, verbose=2)
t_stop = timer()
print("Time to process model of %d words vocabulary for %d Tweets is %0.5f s" % (
    VocLen, len(data_training), (t_stop - t_start)))
print("Accuracy of NeuralNet on validation set is %0.2f" % (100 * score_valid[1]))
print("Accuracy of NeuralNet on test set is %0.2f" % (100 * score_test[1]))


# Save model
emb_model.save("models/emb_model.h5")
emb_model.save_weights("models/emb_weights.h5")

# ====================================
# Plot history
plt.close('all')

AxisLabel = ["Epochs", "Accuracy"]
FName = 'figures/Embedding_accuracy.png'
# FName = None
plot_metric(emb_history, metric_name='accuracy', axis_label=AxisLabel, graph_title="Accuracy plot", file_name=FName)

AxisLabel = ["Epochs", "Loss"]
FName = 'figures/Embedding_Loss.png'
# FName = None
plot_metric(emb_history, metric_name='loss', axis_label=AxisLabel, graph_title="Loss plot", file_name=FName)


# ===================================
# Function for predicting sentiment of sentence
def predict(input_text, thresh=0.5):
    test_seq = tk.texts_to_sequences([input_text])
    test = pad_sequences(test_seq, maxlen=MaxLen)

    s = emb_model.predict(test)
    if s > thresh:
        print("Positive")
    else:
        print("Negative")
    return s


print(predict("You have written a fantastic review about AQM but things have changed."))
print(predict("I have to say your review is nt good."))
print(predict("I have to say your review is not bad."))

# Output in terminal for 20K Voc/Tokenizer
# Epoch 20/20
#  - 5s - loss: 0.4301 - accuracy: 0.8084 - val_loss: 0.4165 - val_accuracy: 0.8162
# 0.80842435
# Time to process model of 20000 words vocabulary for 1600000 Tweets is 109.11987 s
# Accuracy of NeuralNet on validation set is 81.62
# Accuracy of NeuralNet on test set is 79.88
# Positive
# [[0.53042376]]
# Positive
# [[0.5956069]]
# negative
# [[0.17336878]]

# Output in terminal for 10K Voc/Tokenizer
# Epoch 20/20
#  - 5s - loss: 0.4413 - accuracy: 0.8012 - val_loss: 0.3928 - val_accuracy: 0.8357
# 0.8011525
# Time to process model of 10000 words vocabulary for 1600000 Tweets is 104.11279 s
# Accuracy of NeuralNet on validation set is 83.57
# Accuracy of NeuralNet on test set is 79.46
# Positive
# [[0.6165608]]
# Positive
# [[0.6557219]]
# negative
# [[0.18494241]]


# With validation split of 0.01
# Epoch 20/20
#  - 5s - loss: 0.4400 - accuracy: 0.8020 - val_loss: 0.5216 - val_accuracy: 0.7737
# 0.80196273
# Time to process model of 10000 words vocabulary for 1600000 Tweets is 105.70658 s
# Accuracy of NeuralNet on validation set is 84.68
# Accuracy of NeuralNet on test set is 79.72
# Positive
# [[0.72797954]]
# Positive
# [[0.58381724]]
# Negative
# [[0.18854508]]
