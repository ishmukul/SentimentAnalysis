"""
Logistic regression classifier.
It uses CoutVectorizer for creating BagOfWords vectors, where output is compressed matrix of
scipy.sparse.csr.csr_matrix type. This matrix type can be fed directly to LogisticRegression classifier and
thus can work with large data set.

Using 1.6M full training set for classification.
"""

from time import perf_counter as timer
import pandas as pd  # Manipulating data
from sklearn.utils import shuffle  # Shuffle data, in case we need sample of data
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

import heapq

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
# data_training = data_training[:100000]

print("All files loaded")
print("Working with %d tweets" % len(data_training))

# Drop indices for neutral reviews
to_drop = [2]
data_validation = data_validation[~data_validation['Polarity'].isin(to_drop)]
data_test = data_test[~data_test['airline_sentiment'].isin(to_drop)]

# Tweets corresponding to training, validation and test set
tweets_training = data_training['Tweet']
tweets_validation = data_validation['Tweet']
tweets_test = data_test['text']

# Load vocabulary for 1.6M training set. File already created from a previous script
# vocab = pd.read_csv('data/training/vocabulary_training.csv')
# vocab = vocab["Vocabulary"].to_list()
# print(len(vocab))
# print(vocab[:10])

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

t_start = timer()
red_word = 20000  # Number of reduced words in the vocabulary
vocab_final = heapq.nlargest(red_word, vocab, key=vocab.get)  # Reduce vocabulary to red_word
t_stop = timer()
print("Time to create %d words reduced vocabulary for %d Tweets is %0.5f s" % (
    red_word, len(tweets_training), t_stop - t_start))


# Prepare X and y vectors set
def createBagVectors(twt, voc, verbos=False):
    tic = timer()
    vectorizer = CountVectorizer(vocabulary=voc, ngram_range=(1, 2))
    x = vectorizer.fit_transform(twt)
    toc = timer()
    if verbos:
        print("Time to create Doc Vector for BoW for %d Tweets is %0.5f s" % (len(tweets_training), toc - tic))
    return x


print("Working on preparation of X and y sets.")
t_start = timer()
X_train = createBagVectors(tweets_training, vocab_final)
y_train = data_training['Polarity']

X_valid = createBagVectors(tweets_validation, vocab_final)
y_valid = data_validation['Polarity']

X_test = createBagVectors(tweets_test, vocab_final)
y_test = data_test['airline_sentiment']
t_stop = timer()
print("Took %2.4f s for creating X and y vectors")

# =======================================
# Using sklearn classifiers
# Logistic Regression
start_time = timer()
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
Y_pred_log_valid = clf.predict(X_valid)
cm_log_valid = confusion_matrix(y_valid, Y_pred_log_valid)
acc_log_valid = cm_log_valid.trace() / cm_log_valid.sum()
f1_log_valid = f1_score(y_valid, Y_pred_log_valid)
print("Confusion matrix validation set")
print(cm_log_valid)

Y_pred_log_test = clf.predict(X_test)
cm_log_test = confusion_matrix(y_test, Y_pred_log_test)
acc_log_test = cm_log_test.trace() / cm_log_test.sum()
f1_log_test = f1_score(y_test, Y_pred_log_test)
print("Confusion matrix test set")
print(cm_log_test)


end_time = timer()
time_log = end_time - start_time
print('Time for Logistic Regression Validation: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (
    time_log, acc_log_valid, f1_log_valid))
print('Time for Logistic Regression Test: %f s, Accuracy: %0.2f, and F1 score = %0.2f' % (
    time_log, acc_log_test, f1_log_test))


# ===================================
# Function for predicting sentiment of sentence
def predict(text):
    test = createBagVectors([text], vocab_final)

    s = clf.predict(test)
    if s > 0.5:
        print("Positive")
    else:
        print("Negative")
    return s


print(predict("You have written a fantastic review about movie but things have changed."))
print(predict("I have to say your review is not good."))
print(predict("I have to say your review is nt good."))
print(predict("I have to say your review is not bad."))

# Output of Logistic regression
# All files loaded
# Working with 1600000 tweets
# Time to create vocabulary for 1600000 Tweets is 2.53507 s
# Time to create 20000 words reduced vocabulary for 1600000 Tweets is 0.18541 s
# Working on preparation of X and y sets.
# Took %2.4f s for creating X and y vectors
# Confusion matrix validation set
# [[136  41]
#  [ 26 156]]
# Confusion matrix test set
# [[7159 2019]
#  [ 265 2098]]
# Time for Logistic Regression Validation: 87.382610 s, Accuracy: 0.81, and F1 score = 0.82
# Time for Logistic Regression Test: 87.382610 s, Accuracy: 0.80, and F1 score = 0.65
# Positive
# [1]
# Positive
# [1]
# Negative
# [0]
