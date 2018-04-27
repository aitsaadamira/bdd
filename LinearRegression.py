#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:18:02 2018

@author: mira
"""

from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD


def training_set(pos_file, neg_file):
    text_negative = sc.textFile(neg_file)
    text_positive = sc.textFile(pos_file)
    
    train_text = text_negative.union(text_positive)
    train_labels = text_negative.map(lambda x: 0.0).union(text_positive.map(lambda x: 1.0))
    
    tf = HashingTF(numFeatures=10000).transform(train_text.map(lambda x : x))
    idf = IDF().fit(tf)
    train_tfidf = idf.transform(tf)
    
    training = train_labels.zip(train_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
    return (training, idf)


def test_set(test_file, idf):
    test_text = sc.textFile(test_file)
    
    tf_test = HashingTF(numFeatures=10000).transform(test_text.map(lambda x : x))
    tfidf_test = idf.transform(tf_test)
    return tfidf_test


if __name__ == "__main__" :
    
    
    ###########################################################################
    #########                      Spark Context                      #########
    
    conf = SparkConf().\
    setAppName('sentiment-analysis').\
    setMaster('local[*]')
    
    sc = SparkContext(conf = conf)
    
    file = open("resultat_learning.txt","a")
    file.write("\n\n\n\n*******************************************************\n")
    
    
    ###########################################################################
    #########                 Training and Test Set                   #########
    
    pos_file = "data/training_positif_clean.csv"
    neg_file = "data/training_negatif_clean.csv"
    
    training_idf = training_set(pos_file, neg_file)
    training = training_idf[0]
    idf = training_idf[1]
    
    test_file = "data/test_clean" + ".csv"
    test = test_set(test_file, idf)
    
    print("\nDone : Tf-IDF training and test sets")
    
    
    ###########################################################################
    #########                      Model Training                     #########
    
    model_regression = LogisticRegressionWithSGD.train(training)
    print("Done : regression training ")


    ###########################################################################
    #########                     Model Testing                       #########
    
    
    #regression
    predictions_regression = model_regression.predict(test)
    num_pos_regression = predictions_regression.countByValue()[1.0]
    num_neg_regression = predictions_regression.countByValue()[0.0]
    
    print("\n== PREDICTION REGRESSION : ==\n")
    print("- Positive : " , num_pos_regression)
    print("- Negative : " , num_neg_regression)
    
    file.write("\n\n" + "======================== regression ======================== " + "\n\n")
    file.write("- Positive : " + str(num_pos_regression) + "\n")
    file.write("- Negative : " + str(num_neg_regression) + "\n")
    
    
    
    ###########################################################################
    #########           Testing on Brexit Labeled Data                #########
    
    
    print("\n========= Test on Brexit labeled data ========= ")
    
    text_negative_brexit = sc.textFile("data/brexit_negatif_clean.csv")
    text_positive_brexit = sc.textFile("data/brexit_positif_clean.csv")

    test_text_brexit = text_negative_brexit.union(text_positive_brexit)
    test_tlabels_brexit = text_negative_brexit.map(lambda x: 0.0).union(text_positive_brexit.map(lambda x: 1.0))
    
    tf_test_brexit = HashingTF(numFeatures=10000).transform(test_text_brexit.map(lambda x : x))
    
    tfidf_test_brexit = idf.transform(tf_test_brexit)
    
    #prediction and evaluation
    labeled_prediction_regression = test_tlabels_brexit.zip(model_regression.predict(tfidf_test_brexit)).map(lambda x: {"actual": x[0], "predicted": x[1]})
    accuracy_regression = 1.0 * labeled_prediction_regression.filter(lambda doc: doc["actual"] == doc['predicted']).count() / labeled_prediction_regression.count()

    
    print('\n== ACCURACY REGRESSION : ', accuracy_regression , '==')
    
    file.write("\n" + "== Results on labeled data (Brexit) ==" + "\n")
    file.write('\n-> ACCURACY regression : ' + str(accuracy_regression) + '\n')
    
    
    
