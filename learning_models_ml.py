#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:41:07 2018

@author: mira
"""


from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import MultilayerPerceptronClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from datetime import datetime


if __name__ == "__main__":
    
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
    test_text = sc.textFile("data/test_clean.csv")
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
    nb_features = 5000
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
    #########       Tokenization + TF-IDF Brexit Labeled Data         #########
    

    brexit_positive = sc.textFile("data/brexit_positif_clean.csv")
    brexit_negative = sc.textFile("data/brexit_negatif_clean.csv")
    
    pos_labels_brexit = brexit_positive.map(lambda x : 1.0).zip(brexit_positive.map(lambda x : x))
    neg_labels_brexit = brexit_negative.map(lambda x : 0.0).zip(brexit_negative.map(lambda x : x))
    
    pos_df_brexit = pos_labels_brexit.toDF(["label" , "sentence"])
    neg_df_brexit = neg_labels_brexit.toDF(["label" , "sentence"])
    test_df_brexit = pos_df_brexit.union(neg_df_brexit)
    
    tokenizer_test_brexit = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData_test_brexit = tokenizer_test_brexit.transform(test_df_brexit)
    
    hashingTF_test_brexit = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=nb_features)
    featurizedData_test_brexit = hashingTF_test_brexit.transform(wordsData_test_brexit)
    
    rescaledData_test_brexit = idfModel.transform(featurizedData_test_brexit)
    
    rescaled_test_df_brexit = rescaledData_test_brexit.select("features" , "label")
    
    print("\nDone : Tokenization and TF-IDF")
    
    
    ###########################################################################
    #########                     Opening Result File                 #########
    
    file = open("resultat_ml.txt","a")
    file.write("\n\n\n\n********************************************************************************\n")
    file.write(">>> Date time : " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n")
    
    
    ###########################################################################
    #########                    Training and Test                    #########  
    
    print("\n======================================================= ")
    print("==================== NEURAL NETWORK =================== ")
    print("=======================================================\n")
    
    print("\n================== Training ===================\n")
    
    #training model MLP
    num_cols = rescaledData.select('features').collect()[0].features.size  #vocabulary size
    layers = [num_cols , 100 , 2]
    trainer_MLP = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    model_MLP = trainer_MLP.fit(rescaledData)
    print("Done : Neural Network Training")
    
    
    print("\n=================== Testing =================== \n")
    #MLP test
    predictions_MLP = model_MLP.transform(rescaled_test_df)
    #predictions_MLP.show()

    num_pos_mlp = predictions_MLP.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[1.0]
    num_neg_mlp = predictions_MLP.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[0.0]
        
    print("== PREDICTION MLP : ==")
    print("- Positive : " , num_pos_mlp)
    print("- Negative : " , num_neg_mlp)
    
    file.write("\n\n" + "========================= MLP =========================" + "\n\n")
    file.write("- Positive : " + str(num_pos_mlp) + "\n")
    file.write("- Negative : " + str(num_neg_mlp) + "\n")
    
   
    print("\n========= Test on Brexit labeled data =========\n ")

    #MLP
    result_MLP = model_MLP.transform(rescaled_test_df_brexit)
    predictionAndLabels = result_MLP.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy_MLP = evaluator.evaluate(predictionAndLabels)
    print("Accuracy MLP = " + str(accuracy_MLP))
    
        
    file.write("\n" + "== Results on labeled data (Brexit) ==" + "\n")
    file.write('-> ACCURACY MLP : ' + str(accuracy_MLP) + '\n')
    
    
    print("\n======================================================= ")
    print("====================== LINEAR SVC ===================== ")
    print("=======================================================\n")
    
    print("\n================== Training ===================\n")
    
    #training model SVC
    trainer_SVC = LinearSVC(maxIter=10, regParam=0.1)
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

    
    print("\n========= Test on Brexit labeled data =========\n")

    #svc
    result_svc = model_linear_svc.transform(rescaled_test_df_brexit)
    predictionAndLabels = result_svc.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy_SVC = evaluator.evaluate(predictionAndLabels)
    print("Accuracy SVC = " + str(accuracy_SVC))
    
        
    file.write("\n" + "== Results on labeled data (Brexit) ==" + "\n")
    file.write('-> ACCURACY Linear_SVC : ' + str(accuracy_SVC) + '\n')
    
    file.close()













