#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:41:07 2018

@author: mira
"""


from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import LinearSVC
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from datetime import datetime
import sys


if __name__ == "__main__":
    
    part = sys.argv[1]
    
    ###########################################################################
    #########                      Spark Context                      #########
    
    conf = None
    conf = SparkConf().\
    setAppName('sentiment-analysis').\
    setMaster('local[*]')
    
    sc = SparkContext(conf = conf)
    
    spark = SparkSession \
        .builder \
        .appName("ml_classification") \
        .getOrCreate()
        
    
    
    ###########################################################################
    #########        Tokenizing Training and Test Set                #########
    
    
    #test_set
    test_text = sc.textFile("data/test_clean"+ str(part) +".csv")
    test_df = test_text.map(lambda x : (0,x)).toDF(["nothing" , "sentence"]) #(0,x) = bricolage
    
    tokenizer_test = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData_test = tokenizer_test.transform(test_df)
        
    df_test = wordsData_test
    nb_features_test = df_test.rdd.map(lambda x: len(x["words"])).sum()
    
    
    #training set
    text_positive = sc.textFile("data/training_positif_clean.csv")
    text_negative = sc.textFile("data/training_negatif_clean.csv")
    
    pos_labels = text_positive.map(lambda x: 1.0).zip(text_positive.map(lambda x : x))
    neg_labels = text_negative.map(lambda x: 0.0).zip(text_negative.map(lambda x : x))
    
    pos_df = pos_labels.toDF(["label" , "sentence"])
    neg_df = neg_labels.toDF(["label" , "sentence"])
    
    text_df = pos_df.union(neg_df)
    
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(text_df)
     
    df = wordsData
    nb_features_train = df.rdd.map(lambda x: len(x["words"])).sum()

    
    #number of words
    nb_features = max(nb_features_train , nb_features_test)
    print(nb_features)
    nb_features = 10000
    print("\nDone : Tokenization training and test sets")
    
    ###########################################################################
    #########             TF IDF Training and Test Set                #########
    
    #training set
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=nb_features)
    featurizedData = hashingTF.transform(wordsData)
    
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    
    #rescaledData.select("label", "features").show()
    
    #test_set
    hashingTF_test = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=nb_features)
    featurizedData_test = hashingTF_test.transform(wordsData_test)
    
#    idf = IDF(inputCol="rawFeatures", outputCol="features")
#    idfModel = idf.fit(featurizedData_test)
    rescaledData_test = idfModel.transform(featurizedData_test)
    
    rescaled_test_df = rescaledData_test.select("features")
    #rescaledData_test.select("features").show()
    print("\nDone : TF-IDF training and test set")
    

    
    ###########################################################################
    #########                     Opening Result File                 #########
    
    file = open("resultat_ml.txt","a")
    file.write("\n\n\n\n********************************************************************************\n")
    file.write(">>> Date time : " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n")
    file.write(">>> Part : "+ part + " \n")
    
    
    ###########################################################################
    #########                    Training and Test                    #########  

    
    
    print("\n======================================================= ")
    print("====================== LINEAR SVC ===================== ")
    print("=======================================================\n")
    
    print("\n================== Training ===================\n")
    
    #training model SVC
    trainer_SVC = LinearSVC(maxIter=100, regParam=0.1)
    model_linear_svc = trainer_SVC.fit(rescaledData)
    print("Done : Linear_SVC training")
    
    
    print("\n=================== Testing =================== \n")
    
    #SVC test
    predictions_svc = model_linear_svc.transform(rescaled_test_df)
    #predictions_svc.show()
    
    num_pos_svc = predictions_svc.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[1.0]
    num_neg_svc = predictions_svc.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[0.0]
        
    print("\n== PREDICTION SVC : ==")
    print("- Positive : " , num_pos_svc)
    print("- Negative : " , num_neg_svc)
    
    
    file.write("\n\n" + "====================== Linear SVC =====================" + "\n\n")
    file.write("- Positive : " + str(num_pos_svc) + "\n")
    file.write("- Negative : " + str(num_neg_svc) + "\n")

    
    file.close()













