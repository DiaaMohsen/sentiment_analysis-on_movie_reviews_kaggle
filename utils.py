# -*- coding: utf-8 -*-
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

PICKLED_DIR = "pickled_files"

def prepare_data_alittle(text):
    return [re.sub('_|#|:|\n|,|;|«|»|!|-|%|$|@|<|>|\.|\...',' ',t) for t in text]


# c_no: for saving pickled files, on_: which column to label data on, which_v: which vectorizer to build
def blabla(training_df, c_no, on_, which_v=1, model=LinearSVC(C=.1, class_weight="balanced")):
    
    #split data to training and testing
    training_temp, testing_temp = train_test_split(training_df, train_size=.7)    
    
    #preprocess X_train, X_test
    X_train = prepare_data_alittle(training_temp['Phrase'].tolist())
    X_test = prepare_data_alittle(testing_temp['Phrase'].tolist())

    #make Y_train, Y_test labels consist of Sentiment, 3c_sentiment columns 
    Y_train = training_temp.iloc[:, [3,4]].values#training_temp[label_col].tolist()
    Y_test = testing_temp.iloc[:, [3,4]].values#testing_temp[label_col].tolist()
    
    # creat CountVectorizer and if which_v is parameterized then make it TfidfVectorizer
    vectorizer = CountVectorizer(ngram_range=(1,3), stop_words = list(set(stopwords.words("english")) ))
    if which_v != 1:
        vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words = list(set(stopwords.words("english")) ))
    
    
    tr_features = vectorizer.fit_transform(X_train)
    
    
    model = model.fit(tr_features, Y_train[:,on_])
    
    #pickle model and vectorizer to use it later instead of return them 
    pickle.dump(vectorizer, open("%s/vectorizer_%d.pickle" %(PICKLED_DIR, c_no), "wb"))
    pickle.dump(model, open("%s/model_%d.pickle" %(PICKLED_DIR, c_no), "wb"))
    
    
    return X_test, Y_test

def print_report(Y_test, predicted_results, target_names):
    print classification_report(Y_test, predicted_results, target_names=target_names)
    print 'accuracy : ' , accuracy_score(Y_test, predicted_results)

# transform nre record then run classify model-1 if predict -1 then run model-2 elif 1 run model-2
def predict_pipeline(data, model1, v1, model2, v2, model3, v3):
     
    temp1 = v1.transform(data)
    pred = model1.predict(temp1)
    
    for i, t in enumerate(data):
        if pred[i] == -1: #NEGATIVE
            pred[i] = (model2.predict(v2.transform([t]))[0])
        elif pred[i] == 1: #POSITIVE
            pred[i] = (model3.predict(v3.transform([t]))[0])
        else:
            pred[i] =(2)
    
    return pred    
