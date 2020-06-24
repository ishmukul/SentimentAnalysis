"""
A Recurrent Neural Network model using Word embedding training for sentiment analysis.
Embedding layer is trained on full set of 1.6M tweets and 10k word vocabulary.

Model uses two LSTM layers-> Dense layer

Accuracies are pretty good.
Training ~ 81.5%
Validation ~ 81.62%
Test ~ 81.49%

Achievement: Model was able to classify 'not bad' as 'good'.
"""

# Import libraries

import os  # Some OS operations
from time import perf_counter as timer
import pandas as pd  # Manipulating data
from sklearn.utils import shuffle  # Shuffle data, in case we need sample of data

# Keras libraries for Neural Networks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Dropout, Flatten, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Adadelta, RMSprop
import h5py
import pickle
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
# data_training = data_training[:20000]

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
# Some Constants
VocLen = 10000  # Number of words to keep in dictionary in dictionary
MaxLen = 24
time_step = 50  # Same as vector dimension of GloVe Embedding

t_start = timer()  # Clock start

tokenizer = Tokenizer(num_words=VocLen)
tokenizer.fit_on_texts(tweets_training)

# Save token for future use
with open('models/tokenizer_10k.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Load tokenizer
# # Will be used in fast loading this routine.
# with open('models/tokenizer_10k.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# Converting sequences of tweets
X_train_seq = tokenizer.texts_to_sequences(tweets_training)
X_valid_seq = tokenizer.texts_to_sequences(tweets_validation)
X_test_seq = tokenizer.texts_to_sequences(tweets_test)

# Check sequence lengths of all tweets, i.e. length of Tweets' words.
seq_lengths = tweets_training.apply(lambda x: len(x.split(' ')))
# print(seq_lengths.value_counts())

# Every run is random due to shuffling of dataset and selecting 100K tweets. In this run, we have only 3 tweets more
# than 24 length. Ignoringing them and truncating tweets at 24 max length.
# Out of 1.6M tweets nearly 40 reduced tweets have more than 24 length.
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


# Function to define model
def define_model(data):
    modl = Sequential()
    modl.add(Embedding(VocLen, time_step, input_length=data.shape[1]))
    modl.add(Bidirectional(LSTM(64, return_sequences=True))),
    modl.add(Bidirectional(LSTM(32, return_sequences=False))),
    # modl.add(Flatten())
    modl.add(Dense(64, activation='relu'))
    modl.add(Dropout(0.5))
    modl.add(Dense(1, activation='sigmoid'))
    return modl


# Create a model
rnn_emb_model = define_model(X_train)
print(rnn_emb_model.summary())

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

rnn_emb_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

rnn_emb_history = rnn_emb_model.fit(X_train, y_train, epochs=10, batch_size=512, validation_data=[X_valid, y_valid],
                                    verbose=2)
# emb_history = emb_model.fit(X_train, y_train, epochs=20, batch_size=512, validation_split=0.01, verbose=2)
print(rnn_emb_history.history['accuracy'][-1])

score_valid = rnn_emb_model.evaluate(X_valid, y_valid, batch_size=512, verbose=2)
score_test = rnn_emb_model.evaluate(X_test, y_test, batch_size=512, verbose=2)
t_stop = timer()
print("Time to process model of %d words vocabulary for %d Tweets is %0.5f s" % (
    VocLen, len(data_training), (t_stop - t_start)))
print("Accuracy of NeuralNet on validation set is %0.2f" % (100 * score_valid[1]))
print("Accuracy of NeuralNet on test set is %0.2f" % (100 * score_test[1]))

# Save model
rnn_emb_model.save("models/rnn_emb_model.h5")
rnn_emb_model.save_weights("models/rnn_emb_weights.h5")

# ====================================
# Plot history
plt.close('all')

AxisLabel = ["Epochs", "Accuracy"]
FName = 'figures/RNN_Embedding_accuracy.png'
# FName = None
plot_metric(rnn_emb_history, metric_name='accuracy', axis_label=AxisLabel, graph_title="Accuracy plot", file_name=FName)

AxisLabel = ["Epochs", "Loss"]
FName = 'figures/RNN_Embedding_Loss.png'
# FName = None
plot_metric(rnn_emb_history, metric_name='loss', axis_label=AxisLabel, graph_title="Loss plot", file_name=FName)


# ===================================
# Function for predicting sentiment of sentence
def predict(input_text, thresh=0.5):
    text = text_clean(input_text)
    test_seq = tokenizer.texts_to_sequences(text)
    test = pad_sequences(test_seq, maxlen=MaxLen)

    s = rnn_emb_model.predict(test)
    if s > thresh:
        print("Positive")
    else:
        print("Negative")
    return s


print(predict("You have written a fantastic review about movie but things have changed."))
print(predict("I have to say your review is nt good."))
print(predict("I have to say your review is not bad."))

# Train on 1600000 samples, validate on 359 samples
# Epoch 1/10
# 2020-05-27 07:46:53.169677: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
#  - 155s - loss: 0.4967 - accuracy: 0.7541 - val_loss: 0.3822 - val_accuracy: 0.8189
# Epoch 2/10
#  - 156s - loss: 0.4456 - accuracy: 0.7938 - val_loss: 0.3783 - val_accuracy: 0.8245
# Epoch 3/10
#  - 157s - loss: 0.4346 - accuracy: 0.8003 - val_loss: 0.3781 - val_accuracy: 0.8329
# Epoch 4/10
#  - 158s - loss: 0.4279 - accuracy: 0.8042 - val_loss: 0.3820 - val_accuracy: 0.8134
# Epoch 5/10
#  - 158s - loss: 0.4233 - accuracy: 0.8070 - val_loss: 0.3731 - val_accuracy: 0.8189
# Epoch 6/10
#  - 158s - loss: 0.4196 - accuracy: 0.8089 - val_loss: 0.3650 - val_accuracy: 0.8134
# Epoch 7/10
#  - 156s - loss: 0.4167 - accuracy: 0.8106 - val_loss: 0.3732 - val_accuracy: 0.8189
# Epoch 8/10
#  - 159s - loss: 0.4139 - accuracy: 0.8122 - val_loss: 0.3748 - val_accuracy: 0.8050
# Epoch 9/10
#  - 160s - loss: 0.4114 - accuracy: 0.8139 - val_loss: 0.3801 - val_accuracy: 0.8162
# Epoch 10/10
#  - 157s - loss: 0.4089 - accuracy: 0.8151 - val_loss: 0.3731 - val_accuracy: 0.8162
# 0.81509
# Time to process model of 10000 words vocabulary for 1600000 Tweets is 1575.41285 s
# Accuracy of NeuralNet on validation set is 81.62
# Accuracy of NeuralNet on test set is 80.92
# Positive
# [[0.8607207]]
# Negative
# [[0.36957595]]
# Positive
# [[0.75851214]]
