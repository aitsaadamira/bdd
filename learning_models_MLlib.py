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
from pyspark.mllib.tree import DecisionTree
import sys


def training_set(pos_file, neg_file):
    text_negative = sc.textFile(neg_file)
    text_positive = sc.textFile(pos_file)
    
    train_text = text_negative.union(text_positive)
    train_labels = text_negative.map(lambda x: 0.0).union(text_positive.map(lambda x: 1.0))
    
    tf = HashingTF().transform(train_text.map(lambda x : x))
    idf = IDF().fit(tf)
    train_tfidf = idf.transform(tf)
    
    training = train_labels.zip(train_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
    return (training, idf)


def test_set(test_file, idf):
    test_text = sc.textFile(test_file)
    
    tf_test = HashingTF().transform(test_text.map(lambda x : x))
#    idf_test = IDF().fit(tf_test)
    tfidf_test = idf.transform(tf_test)
    return tfidf_test


if __name__ == "__main__" :

#    if (len(sys.argv)) == 3 :
#        print("\nARGS : training_positive, training_negative, test_set")
#        sys.exit()

    ###########################################################################
    #########                      Spark Context                      #########

    conf = SparkConf().\
    setAppName('sentiment-analysis').\
    setMaster('local[*]')
    
    sc = SparkContext(conf = conf)
    
    
    ###########################################################################
    #########                      Training Set                       #########
    
#    "/home/mira/TAF/projet_BDD/code_BDD/train_positif.csv"
#    "/home/mira/TAF/projet_BDD/code_BDD/train_negatif.csv"
#    "/home/mira/TAF/projet_BDD/code_BDD/test.csv"
    
    pos_file = "data/training_positif.csv" #sys.argv[1]
    neg_file = "data/training_negatif.csv" #sys.argv[2]
    
    training_idf = training_set(pos_file, neg_file)
    training = training_idf[0]
    idf = training_idf[1]
    
    #nb_cols = len(training.collect()[0].features.toArray())
    
    ##################
    # SAVE TRAINING !!
    ##################
    
    ###########################################################################
    #########                     Model Training                      #########
    
    
    model_bayes = NaiveBayes.train(training)
    model_bayes.save(sc, "bayes_model_mira.model")
    #save bayes_model
    model_decision_tree_entropy = DecisionTree.trainClassifier(training, categoricalFeaturesInfo={}, impurity="entropy", maxDepth=5, numClasses=2)
    model_decision_tree_entropy.save(sc, "DT_entropy_model_mira.model")
    #save TR_entropy_model
    model_decision_tree_gini = DecisionTree.trainClassifier(training, categoricalFeaturesInfo={}, impurity="gini", maxDepth=5, numClasses=2)
    model_decision_tree_gini.save(sc, "DT_gini_model_mira.model")
    #save TR_gini_model
    

    ###########################################################################
    #########                     Model Testing                       #########
    
    test_file = "data/test.csv" #sys.argv[3]
    test = test_set(test_file, idf)
    
    ##################
    # SAVE TEST !!
    ##################
    
    
    model_bayes = NaiveBayes.load(sc, "bayes_model_mira.model")
    model_decision_tree_entropy = DecisionTree.load(sc, "DT_entropy_model_mira.model")
    model_decision_tree_gini = DecisionTree.load(sc, "DT_gini_model_mira.model")
    
    file = open("resultat_MLlib.txt","a")
    
    #Bayes
    predictions_bayes = model_bayes.predict(test)
    num_pos_bayes = predictions_bayes.countByValue()[1.0]
    num_neg_bayes = predictions_bayes.countByValue()[0.0]
    
    print("========== PREDICTION BAYES : ==========")
    print("- Positive : " , num_pos_bayes)
    print("- Negative : " , num_neg_bayes)
    
    file.write("\n" + "====== NaiveBayes ======" + "\n")
    file.write("- Positive : " + str(num_pos_bayes) + "\n")
    file.write("- Negative : " + str(num_neg_bayes) + "\n")
    
    #decision tree entropy
    predictions_decision_tree_enptropy = model_decision_tree_entropy.predict(test)
    num_pos_entropy = predictions_decision_tree_enptropy.countByValue()[0.0]
    num_neg_entropy = predictions_decision_tree_enptropy.countByValue()[1.0]
    
    #decision tree gini
    print("========== PREDICTION ENTROPY : ==========")
    print("- Positive : " , num_pos_entropy)
    print("- Negative : " , num_neg_entropy)
    
    file.write("\n" + "====== DecisionTree (Entropy) ======" + "\n")
    file.write("- Positive : " + str(num_pos_entropy) + "\n")
    file.write("- Negative : " + str(num_pos_entropy) + "\n")
    
    predictions_decision_tree_gini = model_decision_tree_gini.predict(test)
    num_pos_gini = predictions_decision_tree_gini.countByValue()[0.0]
    num_neg_gini = predictions_decision_tree_gini.countByValue()[1.0]
    
    print("========== PREDICTION GINI : ==========")
    print("- Positive : " , num_pos_gini)
    print("- Negative : " , num_neg_gini)
    
    
    file.write("\n" + "====== DecisionTree (Gini) ======" + "\n")
    file.write("- Positive : " + str(num_pos_gini) + "\n")
    file.write("- Negative : " + str(num_pos_gini) + "\n")
    
    
    
    ###########################################################################
    #########           Testing on Brexit Labeled Data                #########
    
    text_negative = sc.textFile("data/brexit_negatif.csv")
    text_positive = sc.textFile("data/brexit_positif.csv")

    test_text = text_negative.union(text_positive)
    test_tlabels = text_negative.map(lambda x: 0.0).union(text_positive.map(lambda x: 1.0))
    
    tf_test = HashingTF().transform(test_text.map(lambda x : x))
    
    tfidf_test = idf.transform(tf_test)
    
    #prediction and evaluation
    #bayes
    labeled_prediction_bayes = test_tlabels.zip(model_bayes.predict(tfidf_test)).map(lambda x: {"actual": x[0], "predicted": x[1]})
    accuracy_bayes = 1.0 * labeled_prediction_bayes.filter(lambda doc: doc["actual"] == doc['predicted']).count() / labeled_prediction_bayes.count()
    
    #decision tree entropy
    labeled_prediction_entropy = test_tlabels.zip(model_decision_tree_entropy.predict(tfidf_test)).map(lambda x: {"actual": x[0], "predicted": x[1]})
    accuracy_entropy = 1.0 * labeled_prediction_entropy.filter(lambda doc: doc["actual"] == doc['predicted']).count() / labeled_prediction_entropy.count()
    
    #decision tree gini
    labeled_prediction_gini = test_tlabels.zip(model_decision_tree_gini.predict(tfidf_test)).map(lambda x: {"actual": x[0], "predicted": x[1]})
    accuracy_gini = 1.0 * labeled_prediction_gini.filter(lambda doc: doc["actual"] == doc['predicted']).count() / labeled_prediction_gini.count()
    
    print('\n===== ACCURACY BAYES : ', accuracy_bayes , '=====\n')
    print('\n===== ACCURACY DT ENTROPY : ', accuracy_entropy , '=====\n')
    print('\n===== ACCURACY DT GINI : ', accuracy_gini , '=====\n')
    
    
    file.write("\n" + "====== Results on labeled data (Brexit) ======" + "\n")
    file.write('\n===== ACCURACY BAYES : ' + str(accuracy_bayes) + '=====\n')
    file.write('\n===== ACCURACY DT ENTROPY : ' + str(accuracy_entropy) + '=====\n')
    file.write('\n===== ACCURACY DT GINI : ' + str(accuracy_gini) + '=====\n')
    
    file.close()
    
    
    
    
    
    
    
    
    
    
    
    
    