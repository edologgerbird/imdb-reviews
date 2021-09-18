# IMDB Reviews Sentiment Classifier

Repository for the IMDB Reviews Sentiment Classifier
- Developed a LSTM Deep Learning model to classfify the sentiment of a given movie review (positive/negative)
- Built a vocabulary using Word2Vec Model to identify relationships between words.
- Built a Sequential Deep Learning model containing Embedding, Dropout, Convolutional, Maxpooling, LSTM and dense layers.
- Model Accuracy on Test Data: 0.904
 
## Codes and Resources Used

**Python Version:** 3.9

**Packages:** pandas, numpy, sklearn, matplotlip, gensim, nltk, tensorflow, pickle

**Data Set:** https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

- https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
- https://github.com/aptlo10/-Sentiment-Analysis-on-Movie-Reviews/blob/master/Sentiment_Analysison_MovieReviews_15-2-2018.ipynb
- https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis
- https://www.kaggle.com/sravyaysk/twitter-sentiment-prediction-using-lstm

## Exploratory Data Analysis

- Sanity checked for null inputs
- Investigated distribution of labels

## Data Preparation 

- Removing "@" usernames, URLS, and non-alphanumeric characters
- Removed Stopwords
- Stemmed words to their root form using SnowballStemmer
- Split training and testing sets
- Built a Word2Vec Vocab Model using the train set to learn relationships between words
- Vectorised the text corpus
- Padded the sequences to prepare for Deep Learning 
- Built an embedding matrix which contains the weights of each individual word that will be trained

## Model Building

- Built a model with multiple layers including an Embedding layer, Convolutional layer, Maxpooling and LSTM.
- Model Callbacks: included ReduceLROnPlateau to automatically adjust Learning Rate based on gradient and EarlyStopping to reduce overfitting.

## Model Training

- Trained using batch_size 1024, epochs = 32, validation split = 0.1

## Model Performance

- ACCURACY: 0.9039999842643738
- LOSS: 0.26417699456214905

 precision    recall  f1-score   support

    negative       0.91      0.90      0.90      4979
    positive       0.90      0.91      0.90      5021

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
