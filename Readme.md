# Sentiment Analysis of Tweets  

A sentiment analyser model built as a part of project at [AQM](http://www.aqm.io). Idea was to train the model on 
the [Sentiment140](http://help.sentiment140.com/for-students) dataset, and test on a Kaggle Airlie tweets dataset. 
Using different tweets set helps in understanding if the model will work on domain specific tweets or not.  

Separate scripts were created for different models (details below), starting from a basic Logistic Regression model to 
a complex word-embedding trained Recurrent Neural Network. Accuracy for all the models was fairly good, however, a
 test case was build to check the outputs of the model.  
 
The model parameters were further saved to load the model and perform sentiment analysis.   

**Future prospects:**  
1) Improve model at character level encoding.  
2) Build a flask/django based web application to perform sentiment analysis.  
3) Try more pre-trained word embedding models.  
4) Try BERT or its variations to get better model.     

## Folder structure:   
1) **Root**: Main folder containing Readme and scripts.  
2) **data**: Data folder containing:   
	a) train: Training data from [Sentiment140](http://help.sentiment140.com/for-students).  
	b) validation: Test/Validation set from Sentiment140 website.  
	c) test: Test data from Airline tweets on [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment).  
3) **corpora**: Folder for nltk data.      
4) **figures**: All generated figures from the scripts.  
5) **models**: Keras trained models stored in this folder.  

## Files  

**[data_clean.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/data_clean.py)** : Script for pre-processing data.  
**[Embedding.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/Embedding.py)**: Word Embedding model.     
**[NeuralNet.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/NeuralNet.py)**: A 3layer Neural Network model.     
**[LogisticReg.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/LogisticReg.py)**: Logistic regression classifier from sklearn.  
**[RNN_Embedding.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/RNN_Embedding.py)**: A Recurrent Neural Network model using LSTM layers.     
**[Sentiment_analysis.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/Sentiment_analysis.py)**: Sentiment analysis by loading model.  

## File descriptions    
=======================================================  
**[data_clean.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/data_clean.py)**: 
**This script is required to run once for all other scripts.**  

*Uses cleaning function from Ulas' script.*    
It cleans tweets for any stop word and punctuations and dump the cleaned data into training/training_tweets_clean.csv.  
Output file size for 1.6M tweets reduces to ~77MB

This step is important to create a training file to save pre-processing time for every time kernel restarts.
It takes close to 45s for cleaning tweets using this script.   

This script also saves Keras Tokenizer handle.  

=======================================================  
**[LogisticReg.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/LogisticReg.py)**:  
Logistic regression classifier using sklearn modules. trained on full 1.6M tweets.  

It uses CountVectorizer for creating BagOfWords vectors, where output is compressed matrix of
scipy.sparse.csr.csr_matrix type. This matrix type can be fed directly to LogisticRegression classifier and
thus can work with large data set.

Validation accuracy is good but test accuracy was worse (62%).  

=======================================================  
**[NeuralNet.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/NeuralNet.py)**:   
A multi layer dense neural network for sentiment classification.  
Using BagofWords approach, therefore limited to 100K tweets for memory issues.

Accuracies are pretty good.  
Training ~ 83%  
Validation ~ 79%  
Test ~ 76%  
But could not relate between combination of neg-pos words.  

Cases of Overfitting exists.  
Loss function and Accuracies are plotted in Figures:  
[NN_accuracy.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/NN_accuracy.png)    
[NN_loss.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/NN_loss.png)  
<img src="https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/NN_accuracy.png" alt="NN accuracy" width="200"/>
<img src="https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/NN_loss.png" alt="NN Loss" width="200"/>  


=======================================================  
**[Embedding.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/Embedding.py)**:   
A model using Word embedding training for sentiment analysis.  
Embedding layer is trained on full set of 1.6M tweets and 10k word vocabulary.  

Accuracies are pretty good.  
Training ~ 83%  
Validation ~ 79%  
Test ~ 76%  

Changing vocabulary size is not affecting accuracies. Tried 20K also but model starts overfitting.  
Loss function and Accuracies are plotted in Figures:  
[Embedding_accuracy_10KVoc.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/Embedding_accuracy_10KVoc.png)    
[Embedding_loss_10KVoc.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/Embedding_Loss_10KVoc.png)  
[Embedding_accuracy_20KVoc.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/Embedding_accuracy_20KVoc.png)  
[Embedding_loss_20KVoc.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/Embedding_Loss_20KVoc.png)   
<img src="https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/Embedding_accuracy_10KVoc.png" alt="Embedding accuracy" width="200"/> 
<img src="https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/Embedding_Loss_10KVoc.png" alt="Embedding Loss" width="200"/>  



=======================================================  
**[RNN_Embedding.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/RNN_Embedding.py)**:  
A Recurrent Neural Network model using LSTM layers. Best model till now. Accuracies are close to 81% on training, 
validation and test sets, which implies negligible overfititng.  

Script saves model, model weights and tokenizer in 'models' folder. These parameters will be used in fast loading and sentiment analysis.  

Loss function and Accuracies are plotted in Figures:  
[RNN_Embedding_accuracy.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/RNN_Embedding_accuracy.png)  
[RNN_Embedding_loss.png](https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/RNN_Embedding_Loss.png)   
<img src="https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/RNN_Embedding_accuracy.png" alt="RNN accuracy" width="200"/> 
<img src="https://github.com/ishmukul/SentimentAnalysis/blob/master/figures/RNN_Embedding_Loss.png" alt="RNN Loss" width="200"/>  


This model can classify between a combination of neg-pos words.   
All other models simple models failed to do so.

Test outputs other than validation/test sets.  
print(predict("You have written a fantastic review about AQM but things have changed."))  
Positive  [[0.7895597]]  

print(predict("I have to say your review is nt good."))  
Negative [[0.39353248]]  

print(predict("I have to say your review is not bad."))  
Positive [[0.869245]]


=======================================================  
**[Sentiment_analysis.py](https://github.com/ishmukul/SentimentAnalysis/blob/master/Sentiment_analysis.py)**:  
Python file for sentiment analysis.  
Loads models.  
predict function can classify a statement passed to it as Positive or Negative.  


