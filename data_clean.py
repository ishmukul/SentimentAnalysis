"""
Script to preprocessing data.
It cleans tweets for any stop word and punctuations and dump the cleaned data into training_tweets_clean.csv.
File size of output file for 1.6M tweets reduces to ~77MB

This step is important to create a training file to save pre-processing time for every time kernel restarts.
It takes close to 45s for running this script.

It also saves vocabulary for all training set.
"""

# Import libraries

import csv
import pickle
import heapq
import os  # Some OS operations
from collections import Counter

import pandas as pd  # Manipulating data
from keras.preprocessing.text import Tokenizer

# All helper functions in utilities.py
from utilities import *

# ===============================================
# -----------------------------------------------
# Read data files
data_training = pd.read_csv('data/training/training.1600000.processed.noemoticon.csv',
                            encoding='latin-1',  # latin-1 works better than UTF-8.
                            header=None,  # There are no headers in the data set
                            names=["Polarity", "idTweet", "Date", "Query", "idUser", "Tweet"])  # Specify column names

data_validation = pd.read_csv('data/validation/testdata.manual.2009.06.14.csv',
                              encoding='latin-1',  # Following same as training file.
                              header=None,  # There are no headers in the data set
                              names=["Polarity", "idTweet", "Date", "Query", "idUser", "Tweet"])  # Specify column names

data_test_airline = pd.read_csv('data/test/Tweets.csv')
data_test_translink = pd.read_csv('data/test/Translink_tweets.csv')

# ===============================================
# check Polarities and clean data
check_polarity(data_training, "Polarity")
tweets_training = data_clean(data_training, "Tweet")  # data_clean function in utilities.py
# Replace Tweets column with cleaned tweets.
data_training = data_training.assign(Tweet=pd.Series(tweets_training))
data_training = data_training[["Polarity", "Tweet"]]  # We only need Tweets and their corresponding polarities
data_training["Polarity"][data_training["Polarity"] == 4] = 1

if not (os.path.isdir("data/training")):
    os.mkdir("data/training")
# Save cleaned tweets words and polarity in an external file.
# Running this part alone everytime takes 45s on my machine.
data_training.to_csv("data/training/training_tweets_clean.csv", index=False, quoting=csv.QUOTE_NONE)
print(data_training.head())

# ===============================================
# Work on validation set. This is test set in Sentiment140 data
check_polarity(data_validation, "Polarity")
tweets_validation = data_clean(data_validation, "Tweet")   # data_clean function in utilities.py
data_validation = data_validation.assign(Tweet=pd.Series(tweets_validation))
data_validation = data_validation[["Polarity", "Tweet"]]  # We only need Tweets and their corresponding polarities
data_validation["Polarity"][data_validation["Polarity"] == 4] = 1
if not (os.path.isdir("data/validation")):
    os.mkdir("data/validation")
data_validation.to_csv("data/validation/validation_tweets_clean.csv", index=False, quoting=csv.QUOTE_NONE)
print(data_validation.head())

# ===============================================
# Work on Airline test set. This is test set from kaggle
check_polarity(data_test_airline, "airline_sentiment")
tweets_test = data_clean(data_test_airline, "text")   # data_clean function in utilities.py
data_test_airline = data_test_airline.assign(text=pd.Series(tweets_test))
data_test_airline = data_test_airline[["airline_sentiment", "text"]]  # We only need Tweets and their corresponding polarities
data_test_airline["airline_sentiment"][data_test_airline["airline_sentiment"] == "neutral"] = 2
data_test_airline["airline_sentiment"][data_test_airline["airline_sentiment"] == "positive"] = 1
data_test_airline["airline_sentiment"][data_test_airline["airline_sentiment"] == "negative"] = 0

if not (os.path.isdir("data/test")):
    os.mkdir("data/test")
data_test_airline.to_csv("data/test/test_tweets_clean.csv", index=False, quoting=csv.QUOTE_NONE)
print(data_test_airline.head())

# ===============================================
# Work on Translink test set. This is test set from kaggle
check_polarity(data_test_translink, "Sentiment")
tweets_test_translink = data_clean(data_test_translink, "TweetText")   # data_clean function in utilities.py
data_test_translink = data_test_translink.assign(TweetText=pd.Series(tweets_test_translink))
data_test_translink = data_test_translink[["Sentiment", "TweetText"]]  # We only need Tweets and their corresponding polarities
data_test_translink["Sentiment"][data_test_translink["Sentiment"] == "Positive"] = 1
data_test_translink["Sentiment"][data_test_translink["Sentiment"] == "Negative"] = 0

if not (os.path.isdir("data/test")):
    os.mkdir("data/test")
data_test_translink.to_csv("data/test/test_tweets_translink_clean.csv", index=False, quoting=csv.QUOTE_NONE)
print(data_test_translink.head())
# ===============================
# Preparing for inputs for embedding model
# Some Constants
VocLen = 10000  # Number of words to keep in dictionary in dictionary

tokenizer = Tokenizer(num_words=VocLen)
tokenizer.fit_on_texts(tweets_training)

# Save token for future use
with open('models/tokenizer_10k.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ===============================================
# -----------------------------------------------
# Create vocabulary file for 1.6M tweets and dump it for easy read out
t_start = timer()
flat_list = [word for tweet in tweets_training for word in tweet.split()]  # Equivalent of looping over different terms
vocab = Counter()  # Using Counter to create keys and frequency
vocab.update(flat_list)
t_stop = timer()
print("Time to create vocabulary for %d Tweets is %0.5f s" % (len(tweets_training), t_stop - t_start))

# Dump file into a csv for readout
pd.Series([i for i in vocab.keys()], name="Vocabulary").to_csv("data/training/vocabulary_training.csv", index=False)

# Create a short dictionary with more frequent words out of 1.6M tweets
t_start = timer()
red_word = 10000  # Number of reduced words in the vocabulary
vocab_final = heapq.nlargest(red_word, vocab, key=vocab.get)  # Reduce vocabulary to red_word
t_stop = timer()
print("Time to create %d words reduced vocabulary for %d Tweets is %0.5f s" % (
    red_word, len(tweets_training), t_stop - t_start))
# Dump file into a csv for readout
pd.Series(vocab_final, name="Vocabulary").to_csv("data/training/vocabulary_training_final.csv", index=False)

# ===============================================
# -----------------------------------------------
# Plot Word Cloud
# plot_wordcloud(flat_list, "figures/WordCloud_total.png")
# plot_wordcloud(vocab_final, "figures/WordCloud_reduced.png")

# Output
# Backend Qt5Agg is interactive backend. Turning interactive mode on.
# Positive Tweets:  800000
# Neutral Tweets: 0
# Negative Tweets: 800000
# Time for cleaning tweets 48.96057317800296
#    Polarity                                              Tweet
# 0         0  awww thats bummer shoulda got david carr third...
# 1         0  upset cant update facebook texting might cry r...
# 2         0  dived many times ball managed save rest go bou...
# 3         0                  whole body feels itchy like fire
# 4         0                   no not behaving im mad cant see
# Positive Tweets:  182
# Neutral Tweets: 139
# Negative Tweets: 177
# Time for cleaning tweets 0.012214083995786496
#    Polarity                                              Tweet
# 0         1    loooooooovvvvvveee not dx cool fantastic right
# 1         1                 reading love lee childs good read
# 2         1                  ok first assesment fucking rocks
# 3         1  youll love ive mine months never looked back n...
# 4         1                         fair enough think perfect
# Positive Tweets:  0
# Neutral Tweets: 0
# Negative Tweets: 0
# Time for cleaning tweets 0.4056314879999263
#   airline_sentiment                                               text
# 0                 2                                              said
# 1                 1     plus youve added commercials experience tacky
# 2                 2      didnt today must mean need take another trip
# 3                 0  really aggressive blast obnoxious entertainmen...
# 4                 0                              really big bad thing
# Time to create vocabulary for 1600000 Tweets is 1.96493 s
# Time to create 10000 words reduced vocabulary for 1600000 Tweets is 0.25601 s
