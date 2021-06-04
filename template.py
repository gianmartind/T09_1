# -*- coding: utf-8 -*-
"""
Created on Mon May 31 09:56:56 2021

@author: MVC
"""

import re
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# full data frame
DF = 0
# train df
DF_TRAIN = 0
# test df
DF_TEST = 0
# binary word count vectorizer
CVEC = 0
# tf idf vectorizer
TVEC = 0
# logistic regressor
LOG_REG = 0

def readFile(fileName):
    # read file as df, assign to global var DF
    ## no return val
    global DF
    DF = pd.read_csv(fileName)

def txtCleaner(txt):
    # clean text (String), return cleaned text
    ## this function will be used in dfCleaner
    text = re.sub('[^A-Za-z ]', '', txt)
    text = re.sub(' +', ' ', text)
    return text.lower()

def dfCleaner():
    ## cleaning text in df's review-column
    
    global DF
    for idx, item in DF.iterrows():
        ## replace review text in particular df's row with cleaned text
        DF.at[idx,'review'] = txtCleaner(item.review)
    ## no return val

def splitDf():
    ## splitting DF into train & test dataset, no return val

    global DF_TRAIN, DF_TEST
    
    ## split positive & negative reviews
    df_pos = DF.loc[DF['sentiment'] == 'positive']
    df_neg = DF.loc[DF['sentiment'] == 'negative']
    
    ## get sample rows (from positive reviews) for training using pandas function sample()
    df_pos_train = df_pos.sample(frac=0.7)
    ## get the rest of rows for testing
    df_pos_test = df_pos.drop(df_pos_train.index)
    
    # do train & test splitting for negative reviews
    df_neg_train = df_neg.sample(frac=0.7)
    df_neg_test = df_neg.drop(df_neg_train.index)
    
    # merge positive and negative dfs
    DF_TRAIN = df_pos_train.append(df_neg_train)
    DF_TEST = df_pos_test.append(df_neg_test)


def vectCnt():
    ## binary events vectorizer, return X for train and X for test
    
    global CVEC, DF_TRAIN, DF_TEST
    
    CVEC = CountVectorizer(binary=True)
    
    CVEC.fit(DF_TRAIN['review'].values)

    X = CVEC.transform(DF_TRAIN['review'].values) # fit_transform DF_TRAIN review data using function in object CVEC
    X_test = CVEC.transform(DF_TEST['review'].values) # transform review data in DF_TEST, using vectorizer model in CVEC
    
    return X, X_test

def vectTFIDF():
    ## TF-IDF vectorizer, return X for train and X for test
    
    global TVEC, DF_TRAIN, DF_TEST

    TVEC = TfidfVectorizer(binary=True)
    ## TVEC: TF-IDF Vectorizer 
    TVEC.fit(DF_TRAIN['review'].values)
    # your code here, similar with vectCnt
    X = TVEC.transform(DF_TRAIN['review'].values) # fit_transform DF_TRAIN review data using function in object CVEC
    X_test = TVEC.transform(DF_TEST['review'].values) # transform review data in DF_TEST, using vectorizer model in CVEC
    
    return X, X_test
    
def classifier():
    ## logistic regression classifier
    global DF_TRAIN, DF_TEST
    global LOG_REG
    
    ## X train and test from binary events vectorizer
    XW_train, XW_test= vectCnt()
    ## X train and test from tf-idf vectorizer
    XT_train, XT_test= vectTFIDF()
    ## y train and test from labels in df train and df test
    y_train, y_test = DF_TRAIN['sentiment'], DF_TEST['sentiment']
    
    ## logistic regression object, with inverse of regularization strength = 0.05
    LOG_REG = LogisticRegression(C=0.05)
    
    ## build model from train data (which used binary vector model)
    LOG_REG.fit(XW_train, y_train)
    ## fit test data into model, calculate accuracy
    print ("Accuracy WC: %s" 
           % (accuracy_score(y_test, LOG_REG.predict(XW_test))))
    
    # build model from train data (which used TF-IDF vector model)
    LOG_REG.fit(XT_train, y_train)
    # fit test data into model and calculate accuracy
    print ("Accuracy TF-IDF: %s" 
           % (accuracy_score(y_test, LOG_REG.predict(XT_test))))
    
def sentWordList():
    ## get sentiment word list from classification result
    
    global LOG_REG
    global CVEC # use better vectorizer model (CVEC / TVEC)
    
    ## negative word list
    neg_list = []
    ## positive word list
    pos_list = []
    
    ## make feature:coef dictionary, round coef to 3 decimals
    # your code here, see example from the article
    
    ## If you still confused, see what's inside feature_to_coef by removing # below
    #print(feature_to_coef)
    
    ## fill in the pos & neg list
    # your code here, itterate through items in feature_to_coef dictionary and check coef value
    # if coef value < 0: item goes to neg_list, elif coef value > 0: item goes to pos_list

    ## dump sorted positive word list (from the most positive to less) into .pickle file
    pickle.dump(sorted(pos_list, key=lambda x: x[1], reverse=True), open('posList.pickle', 'wb'))
    # dump sorted negative word list (from the most negative to less) into negList.pickle file
    
def main():
    readFile('IMBD Dataset.csv')
    dfCleaner()
    splitDf()
    classifier()
    sentWordList()
    
    ## Use these lines below to check your list 
    #print(pickle.load(open('posList.pickle', 'rb')))
    #print(pickle.load(open('negList.pickle', 'rb')))
    
if __name__ == '__main__':
    main()
