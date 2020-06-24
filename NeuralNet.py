"""
A dense neural network for sentiment classification.
"""

# Import libraries

import os  # Some OS operations
from time import perf_counter as timer
import pandas as pd  # Manipulating data

from sklearn.utils import shuffle  # Shuffle data, in case we need sample of data
from collections import Counter
import heapq
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# Keras libraries for Neural Networks
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Adadelta

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
data_training = shuffle(data_training)  # Shuffle the data
data_training.reset_index(inplace=True, drop=True)
data_training = data_training[:100000]

print("All files loaded")
print("Working with %d tweets", len(data_training))

# ===============================================
# Preparing files for network
# Drop indices for neutral reviews
to_drop = [2]
data_validation = data_validation[~data_validation['Polarity'].isin(to_drop)]
data_test = data_test[~data_test['airline_sentiment'].isin(to_drop)]

# Tweets corresponding to training, validation and test set
tweets_training = data_training['Tweet']
tweets_validation = data_validation['Tweet']
tweets_test = data_test['text']

# Create vocabulary from reduced tweets and dump it for easy read out.
# It takes ~2 s for creating a vocabulary for 1.6M reduced tweets.
t_start = timer()
flat_list = [word for tweet in tweets_training for word in tweet.split()]  # Equivalent of looping over different terms
# vocab = [*set(flat_list), ]  # Using set and converting to a list
vocab = Counter()
vocab.update(flat_list)
t_stop = timer()
print("Time to create vocabulary for %d Tweets is %0.5f s" % (len(tweets_training), t_stop - t_start))
# Output in terminal
# Time to create vocabulary for 1000 Tweets is 0.00122 s
# Time to create vocabulary for 1600000 Tweets is 1.76170 s

# Reducing limits with heapq package
t_start = timer()
red_word = 10000  # Number of reduced words in the vocabulary
vocab_final = heapq.nlargest(red_word, vocab, key=vocab.get)  # Reduce vocabulary to red_word
t_stop = timer()
print("Time to create %d words reduced vocabulary for %d Tweets is %0.5f s" % (
    red_word, len(tweets_training), t_stop - t_start))


# ========================
# Function for creating a bag of vectors/ Dictionary vectors
def createBagVectors(twt, voc, verbos=False):
    tic = timer()
    vectorizer = CountVectorizer(vocabulary=voc, ngram_range=(1, 2))
    x = vectorizer.fit_transform(twt)
    toc = timer()
    if verbos:
        print("Time to create Doc Vector for BoW for %d Tweets is %0.5f s" % (len(tweets_training), toc - tic))
    return x


# ===========================
# Setting X and y fro training, validation and test set
X_train = createBagVectors(tweets_training, vocab_final)
y_train = data_training['Polarity']

X_valid = createBagVectors(tweets_validation, vocab_final)
y_valid = data_validation['Polarity']

X_test = createBagVectors(tweets_test, vocab_final)
y_test = data_test['airline_sentiment']

# ===========================================
# -------------------------------------------
# Neural network training
# Reshape vectors for Neural Network
X_train = X_train.toarray()
X_valid = X_valid.toarray()
X_test = X_test.toarray()

t_start = timer()


# Define a model. Sequential model used in this case
def define_model(data):
    modl = Sequential()
    modl.add(Dense(units=10, activation='relu', input_dim=data.shape[1]))
    modl.add(Dense(units=10, activation='relu'))  # First hidden layer
    modl.add(Dense(units=10, activation='relu'))  # Second hidden layer
    modl.add(Dropout(0.2))  # Dropout layer
    modl.add(Dense(units=1, activation='sigmoid'))  # Output layer
    return modl


# Create a model
nn_model = define_model(X_train)
print(nn_model.summary())

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

# compile model
nn_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

# Fit model
nn_history = nn_model.fit(X_train, y_train, epochs=20, batch_size=1024, validation_data=[X_valid, y_valid], verbose=2)
score_valid = nn_model.evaluate(X_valid, y_valid, batch_size=1024, verbose=2)
score_test = nn_model.evaluate(X_test, y_test, batch_size=1024, verbose=2)
t_stop = timer()
print("Time for Neural Network", (t_stop - t_start))
print("Accuracy of NeuralNet on validation set is %0.2f" % (100 * score_valid[1]))
print("Accuracy of NeuralNet on test set is %0.2f" % (100 * score_test[1]))
# Output in terminal

# Save model
nn_model.save("models/nn_bow_model.h5")
nn_model.save_weights("models/nn_bow_weights.h5")

y_pred_valid = (nn_model.predict(X_test) > 0.5).astype(int)
cm_valid = confusion_matrix(y_valid, y_pred_valid)
acc_valid = cm_valid.trace() / cm_valid.sum()
f1_valid = f1_score(y_valid, y_pred_valid)
print(cm_valid)

y_pred_keras = (nn_model.predict(X_test) > 0.5).astype(int)
cm_keras = confusion_matrix(y_test, y_pred_keras)
acc_keras = cm_keras.trace() / cm_keras.sum()
f1_keras = f1_score(y_test, y_pred_keras)
print(cm_keras)

print('Time for Neural Network validation: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (
    t_stop - t_start, score_test[1], f1_valid))
print('Time for Neural Network test: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (
    t_stop - t_start, score_test[1], f1_keras))


# Output in terminal
# Time for Neural Network 43.696274390000326
# Accuracy of NeuralNet on validation set is 79.67
# Accuracy of NeuralNet on test set is 78.80
# [[7146 2032]
#  [ 415 1948]]
# Time for Keras Neural Network: 43.696274 s, Accuracy: 0.79, and F1 score = 0.61


# ===================================
# Function for predicting sentiment of sentence
def predict(text):
    test = createBagVectors([text], vocab_final)
    s = nn_model.predict(test)
    if s > 0.5:
        print("Positive")
    else:
        print("negative")
    return s


print(predict("You have written a fantastic review about movie but things have changed."))
print(predict("I have to say your review is not good."))
print(predict("I have to say your review is nt good."))
print(predict("I have to say your review is not bad."))

# ====================================
# Plot history

AxisLabel = ["Epochs", "Accuracy"]
FName = 'figures/NN_accuracy.png'
plot_metric(nn_history, metric_name='accuracy', axis_label=AxisLabel, graph_title="Accuracy plot", file_name=FName)

AxisLabel = ["Epochs", "Loss"]
FName = 'figures/NN_loss.png'
plot_metric(nn_history, metric_name='loss', axis_label=AxisLabel, graph_title="Loss plot", file_name=FName)

# All files loaded
# Working with %d tweets 100000
# Time to create vocabulary for 100000 Tweets is 0.15649 s
# Time to create 10000 words reduced vocabulary for 100000 Tweets is 0.03063 s
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_2 (Dense)              (None, 10)                100010
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                110
# _________________________________________________________________
# dense_4 (Dense)              (None, 10)                110
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 10)                0
# _________________________________________________________________
# dense_5 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 100,241
# Trainable params: 100,241
# Non-trainable params: 0
# _________________________________________________________________
# None
# Train on 100000 samples, validate on 359 samples
# Epoch 1/20
#  - 7s - loss: 0.6295 - accuracy: 0.6631 - val_loss: 0.5014 - val_accuracy: 0.7855
# Epoch 2/20
#  - 4s - loss: 0.5195 - accuracy: 0.7561 - val_loss: 0.4549 - val_accuracy: 0.8022
# Epoch 3/20
#  - 4s - loss: 0.4885 - accuracy: 0.7759 - val_loss: 0.4355 - val_accuracy: 0.7911
# Epoch 4/20
#  - 4s - loss: 0.4667 - accuracy: 0.7898 - val_loss: 0.4447 - val_accuracy: 0.7911
# Epoch 5/20
#  - 4s - loss: 0.4538 - accuracy: 0.7982 - val_loss: 0.4245 - val_accuracy: 0.8050
# Epoch 6/20
#  - 4s - loss: 0.4404 - accuracy: 0.8054 - val_loss: 0.4162 - val_accuracy: 0.8162
# Epoch 7/20
#  - 4s - loss: 0.4311 - accuracy: 0.8113 - val_loss: 0.4240 - val_accuracy: 0.8050
# Epoch 8/20
#  - 4s - loss: 0.4189 - accuracy: 0.8190 - val_loss: 0.4315 - val_accuracy: 0.8106
# Epoch 9/20
#  - 4s - loss: 0.4104 - accuracy: 0.8222 - val_loss: 0.4326 - val_accuracy: 0.7967
# Epoch 10/20
#  - 4s - loss: 0.4002 - accuracy: 0.8282 - val_loss: 0.4372 - val_accuracy: 0.8050
# Epoch 11/20
#  - 4s - loss: 0.3897 - accuracy: 0.8334 - val_loss: 0.4414 - val_accuracy: 0.8134
# Epoch 12/20
#  - 5s - loss: 0.3813 - accuracy: 0.8382 - val_loss: 0.4565 - val_accuracy: 0.8078
# Epoch 13/20
#  - 5s - loss: 0.3725 - accuracy: 0.8435 - val_loss: 0.4585 - val_accuracy: 0.8022
# Epoch 14/20
#  - 4s - loss: 0.3650 - accuracy: 0.8457 - val_loss: 0.4706 - val_accuracy: 0.8106
# Epoch 15/20
#  - 4s - loss: 0.3557 - accuracy: 0.8506 - val_loss: 0.4757 - val_accuracy: 0.8050
# Epoch 16/20
#  - 4s - loss: 0.3469 - accuracy: 0.8553 - val_loss: 0.4928 - val_accuracy: 0.8106
# Epoch 17/20
#  - 4s - loss: 0.3402 - accuracy: 0.8579 - val_loss: 0.5123 - val_accuracy: 0.8050
# Epoch 18/20
#  - 5s - loss: 0.3315 - accuracy: 0.8609 - val_loss: 0.5129 - val_accuracy: 0.8050
# Epoch 19/20
#  - 5s - loss: 0.3249 - accuracy: 0.8642 - val_loss: 0.5361 - val_accuracy: 0.8022
# Epoch 20/20
#  - 5s - loss: 0.3173 - accuracy: 0.8672 - val_loss: 0.5443 - val_accuracy: 0.8106
# Time for Neural Network 91.96976400000005
# Accuracy of NeuralNet on validation set is 81.06
# Accuracy of NeuralNet on test set is 76.67
# [[6842 2336]
#  [ 357 2006]]
# Time for Keras Neural Network: 91.969764 s, Accuracy: 0.77, and F1 score = 0.60
# Positive
# [[0.86887926]]
# Positive
# [[0.7048063]]
# Positive
# [[0.73820853]]