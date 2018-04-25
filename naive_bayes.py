#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:18:02 2018

@author: mira
"""

from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from datetime import datetime
import sys

#/home/mira/TAF/projet_BDD/code_BDD/test_petit_jeu_de_donnees

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
    
    part = sys.argv[1]
    
    
    ###########################################################################
    #########                      Spark Context                      #########
    
    conf = SparkConf().\
    setAppName('sentiment-analysis').\
    setMaster('local[*]')
    
    sc = SparkContext(conf = conf)
    
    file = open("resultat_MLlib.txt","a")
    file.write("\n\n\n\n********************************************************************************\n")
    file.write(">>> Date time : " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n")
    file.write(">>> Part : "+ str(part) + " \n")
    
    
    ###########################################################################
    #########                 Training and Test Set                   #########
    
    pos_file = "data/training_positif_clean.csv"
    neg_file = "data/training_negatif_clean.csv"
    
    training_idf = training_set(pos_file, neg_file)
    training = training_idf[0]
    idf = training_idf[1]
    
    test_file = "data/test_clean"+ str(part) + ".csv"
    test = test_set(test_file, idf)
    
    print("\nDone : Tf-IDF training and test sets")
    
    
    ###########################################################################
    #########                     Model Training                      #########
    
    
    print("\n======================================================= ")
    print("======================== BAYES ======================== ")
    print("=======================================================\n")
    
    print("\n================== Training ===================\n")
    
    model_bayes = NaiveBayes.train(training)
    print("Done : Bayes training ")


    ###########################################################################
    #########                     Model Testing                       #########
    
    print("\n=================== Testing =================== \n")
    
    
    #Bayes
    predictions_bayes = model_bayes.predict(test)
    num_pos_bayes = predictions_bayes.countByValue()[1.0]
    num_neg_bayes = predictions_bayes.countByValue()[0.0]
    
    print("== PREDICTION BAYES : ==\n")
    print("- Positive : " , num_pos_bayes)
    print("- Negative : " , num_neg_bayes)
    
    file.write("\n\n" + "======================== BAYES ======================== " + "\n\n")
    file.write("- Positive : " + str(num_pos_bayes) + "\n")
    file.write("- Negative : " + str(num_neg_bayes) + "\n")
    
    
    
    ###########################################################################
    #########           Testing on Brexit Labeled Data                #########
    
    
    print("\n========= Test on Brexit labeled data ========= ")
    
    text_negative = sc.textFile("data/brexit_negatif_clean.csv")
    text_positive = sc.textFile("data/brexit_positif_clean.csv")

    test_text = text_negative.union(text_positive)
    test_tlabels = text_negative.map(lambda x: 0.0).union(text_positive.map(lambda x: 1.0))
    
    tf_test = HashingTF(numFeatures=10000).transform(test_text.map(lambda x : x))
    
    tfidf_test = idf.transform(tf_test)
#    
#    prediction and evaluation
#    bayes
    labeled_prediction_bayes = test_tlabels.zip(model_bayes.predict(tfidf_test)).map(lambda x: {"actual": x[0], "predicted": x[1]})
    accuracy_bayes = 1.0 * labeled_prediction_bayes.filter(lambda doc: doc["actual"] == doc['predicted']).count() / labeled_prediction_bayes.count()

    
    print('\n== ACCURACY BAYES : ', accuracy_bayes , '==')
    
    file.write("\n" + "== Results on labeled data (Brexit) ==" + "\n")
    file.write('\n-> ACCURACY BAYES : ' + str(accuracy_bayes) + '\n')
    
    

    file.close()
    
    
    
    
    
    