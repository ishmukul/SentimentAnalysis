"""
Helper functions common to all files.
"""
from time import perf_counter as timer

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.data.path.append(".")  # Tell nltk to look into directory mentioned above


# Check number of tweets with a given polarities of given tweets.
def check_polarity(df, name="Polarity"):
    """
    Check number of tweets with a given polarities of given tweets.
    Check how many tweets are positive, negative or neutral
    :param df: dataframe input
    :param name: Polarity column name. Default=Polarity
    :return: none
    """
    polarity = df[name]
    positives = polarity[polarity == 4]
    neutrals = polarity[polarity == 2]
    negatives = polarity[polarity == 0]
    print('Positive Tweets:  {}'.format(len(positives)))
    print('Neutral Tweets: {}'.format(len(neutrals)))
    print('Negative Tweets: {}'.format(len(negatives)))
    return None


# Clean all dataframe
def data_clean(df, name="Tweet"):
    """
    Clean a pandas dataframe containing several tweets for sentiment analysis.
    :param df: Pandas Data frame
    :param name: Pandas tweet column name
    :return: Array of cleaned tweets
    """
    tic = timer()
    twts = []
    # Define a punctuation dictionary so that we can replace each punctuation with an empty space.
    table = str.maketrans('', '', string.punctuation)
    stopWords = set(stopwords.words('senti'))  # Set stop words language to English
    for n in range(df[name].shape[0]):
        text = df[name][n]
        tokens = text.split()  # Split each tweet into list of words.
        tokens = filter(lambda x: x[0] != '@', tokens)  # Remove mentions
        tokens = [word.translate(table) for word in tokens]  # Remove punctuation marks
        tokens = [word for word in tokens if word.isalpha()]  # Remove any word that is not completely alphabetic.
        tokens = [word for word in tokens if len(word) > 1]  # Remove any word that is shorter than two letters
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if not word in stopWords]  # Remove any stopwords
        # Modified for dumping data without additional commas in csv file
        token = ""
        for i in tokens:
            token += (i + " ")
        twts.append(token)
    toc = timer()
    print("Time for cleaning tweets", (toc - tic))
    return twts


# Clean a simple text
def text_clean(text):
    """
    Clean a single line statement for sentiment analysis.
    :param text: Str sentence
    :return: String array
    """
    out = []
    # Define a punctuation dictionary so that we can replace each punctuation with an empty space.
    table = str.maketrans('', '', string.punctuation)
    stopWords = set(stopwords.words('senti'))  # Set stop words language to English
    tokens = text.split()  # Split each tweet into list of words.
    tokens = filter(lambda x: x[0] != '@', tokens)  # Remove mentions
    tokens = [word.translate(table) for word in tokens]  # Remove punctuation marks
    tokens = [word for word in tokens if word.isalpha()]  # Remove any word that is not completely alphabetic.
    tokens = [word for word in tokens if len(word) > 1]  # Remove any word that is shorter than two letters
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in stopWords]  # Remove any stopwords
    token = ""
    for i in tokens:
        token += (i + " ")
    out.append(token)
    return out


def plot_metric(history_name, metric_name='accuracy', axis_label=None, graph_title=None, file_name="", dpi=100,
                xaxis_tick_label=None):
    """
    Function for plotting Neural network metrics, e.g. accuracy curve, loss curve, etc.
    :param history_name: Pointer for model history
    :param metric_name: accuracy, loss
    :param axis_label: [xaxis_label, yaxis_label]
    :param graph_title: Plot title
    :param file_name: Filename for saving file
    :param dpi: figure resolution
    :param xaxis_tick_label: Arbitrary xticks (in case of special names)
    :return: Nothing
    """
    metric = history_name.history[metric_name]
    validation_metric = history_name.history['val_' + metric_name]
    epochs = range(1, len(metric) + 1)
    plt.figure(figsize=plt.figaspect(1.), dpi=dpi)
    plt.plot(epochs, metric, 'bo', label='Training ' + metric_name.capitalize())
    plt.plot(epochs, validation_metric, 'r', label='Validation ' + metric_name.capitalize())
    if axis_label is None:
        axis_label = ['Epochs', 'met']
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    if graph_title is None:
        graph_title = metric_name.capitalize()
    plt.title(graph_title)
    if xaxis_tick_label:
        plt.xticks(epochs, xaxis_tick_label, rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    return


def plot_bar(label_array, acc_array, f1_array, width=0.5, axis_label=None, graph_title=None, file_name="", dpi=100):
    """
    Function for plotting Neural network metrics, e.g. accuracy curve, loss curve, etc.
    :param axis_label: [xaxis_label, yaxis_label]
    :param graph_title: Plot title
    :param file_name: Filename for saving file
    :param dpi: figure resolution
    :return: Nothing
    """
    plt.figure(figsize=plt.figaspect(1.), dpi=dpi)
    x = np.arange(len(label_array))  # the label locations
    plt.bar(x - 0.5 * width, acc_array, width, label='Accuracy')
    plt.bar(x + 0.5 * width, f1_array, width, label='F1 score')
    plt.ylim([0, 1.1])
    plt.xticks(x, labels=label_array)
    if axis_label is None:
        axis_label = ['Set', 'Values']
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    if graph_title is None:
        graph_title = graph_title
    plt.title(graph_title)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    return


# Create a word cloud
def plot_wordcloud(word_list, file_name=""):
    """
    Function to create a Word Cloud
    :param word_list: List of all words
    :param file_name: File name for saving option
    :return: None
    """
    plt.figure(figsize=plt.figaspect(0.8), dpi=100)
    all_words = ' '.join([text for text in word_list])
    wordcloud = WordCloud(width=800, height=600, random_state=21, max_font_size=110).generate(all_words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    return
