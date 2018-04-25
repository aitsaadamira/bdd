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


if __name__ == "__main__":
    
    ###########################################################################
    #########                      Spark Context                      #########
    
    
    conf = SparkConf().\
    setAppName('sentiment-analysis').\
    setMaster('local[*]')
    
    sc = SparkContext(conf = conf)
    
    spark = SparkSession \
        .builder \
        .appName("ml_classification") \
        .getOrCreate()
        
        
    ###########################################################################
    #########                        Training                         #########
    
    
    text_positive = sc.textFile("bdd/data/training_positif.csv")
    text_negative = sc.textFile("bdd/data/training_negatif.csv")
    
    pos_labels = text_positive.map(lambda x: 1.0).zip(text_positive.map(lambda x : x))
    neg_labels = text_negative.map(lambda x: 0.0).zip(text_negative.map(lambda x : x))
    
    pos_df = pos_labels.toDF(["label" , "sentence"])
    neg_df = neg_labels.toDF(["label" , "sentence"])
    
    text_df = pos_df.union(neg_df)
    
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(text_df)
    
    
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    featurizedData = hashingTF.transform(wordsData)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
    
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    
    rescaledData.select("label", "features").show()
    
    
    #training model MLP
    num_cols = rescaledData.select('features').collect()[0].features.size  #vocabulary size
    layers = [num_cols , 500 , 2]
    trainer_MLP = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    model_MLP = trainer_MLP.fit(rescaledData)
    trainer_MLP.save("bdd/model_MLP.model")
    
    
    #training model SVC
    trainer_SVC = LinearSVC(maxIter=10, regParam=0.1)
    model_linear_svc = trainer_SVC.fit(rescaledData)
    trainer_SVC.save("bdd/model_SVC.model")
    # Print the coefficients and intercept for linearsSVC
    print("Coefficients: " + str(model_linear_svc.coefficients))
    print("Intercept: " + str(model_linear_svc.intercept))
    

    ###########################################################################
    #########                           Test                          ######### 
    
    test_text = sc.textFile("bdd/data/test.csv")
    test_df = test_text.map(lambda x : x).map(lambda x : (0,x)).toDF(["nothing" , "sentence"]) #(0,x) = bricolage
    
    tokenizer_test = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData_test = tokenizer_test.transform(test_df)
    
    hashingTF_test = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    featurizedData_test = hashingTF_test.transform(wordsData_test)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
    
#    idf = IDF(inputCol="rawFeatures", outputCol="features")
#    idfModel_test = idf.fit(featurizedData_test)
    rescaledData_test = idfModel.transform(featurizedData_test)
    
    rescaled_test_df = rescaledData_test.select("features")
    rescaledData_test.select("features").show()
    
    
    model_MLP = MultilayerPerceptronClassifier.load("bdd/model_MLP.model")
    model_MVC = LinearSVC.load("bdd/model_SVC.model")
    
    file = open("bdd/resultat_ml.txt","a")
    
    #MLP test
    predictions_MLP = model_MLP.transform(rescaled_test_df)
    predictions_MLP.show()

    num_pos_mlp = predictions_MLP.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[1.0]
    num_neg_mlp = predictions_MLP.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[0.0]
        
    print("========== PREDICTION MLP : ==========")
    print("- Positive : " , num_pos_mlp)
    print("- Negative : " , num_neg_mlp)
    
    file.write("\n" + "====== MLP ======" + "\n")
    file.write("- Positive : " + str(num_pos_mlp) + "\n")
    file.write("- Negative : " + str(num_neg_mlp) + "\n")
    
    #SVC test
    predictions_svc = model_linear_svc.transform(rescaled_test_df)
    predictions_svc.show()
    
    num_pos_svc = predictions_svc.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[1.0]
    num_neg_svc = predictions_svc.select("prediction").rdd.map(lambda x : x["prediction"]).countByValue()[0.0]
        
    print("========== PREDICTION : ==========")
    print("- Positive : " , num_pos_svc)
    print("- Negative : " , num_neg_svc)
    
    file.write("\n" + "====== Linear SVC ======" + "\n")
    file.write("- Positive : " + str(num_pos_svc) + "\n")
    file.write("- Negative : " + str(num_neg_svc) + "\n")



    ###########################################################################
    #########              Test On Brexit Labeled Data                #########
    
    
    brexit_positive = sc.textFile("bdd/data/brexit_positif.csv")
    text_negative = sc.textFile("bdd/data/brexit_negatif.csv")
    
    pos_labels = text_positive.map(lambda x : 1.0).zip(text_positive.map(lambda x : x))
    neg_labels = text_negative.map(lambda x : 0.0).zip(text_negative.map(lambda x : x))
    
    pos_df = pos_labels.toDF(["label" , "sentence"])
    neg_df = neg_labels.toDF(["label" , "sentence"])
    test_df = pos_df.union(neg_df)
    
    tokenizer_test = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData_test = tokenizer_test.transform(test_df)
    
    hashingTF_test = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    featurizedData_test = hashingTF_test.transform(wordsData_test)
    
    rescaledData_test = idfModel.transform(featurizedData_test)
    
    rescaled_test_df = rescaledData_test.select("features" , "label")
    
    #MLP
    result = model_MLP.transform(rescaled_test_df)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy_MLP = evaluator.evaluate(predictionAndLabels)
    print("Accuracy MLP = " + str(accuracy_MLP))
    
    #svc
    result = model_linear_svc.transform(rescaled_test_df)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy_SVC = evaluator.evaluate(predictionAndLabels)
    print("Accuracy SVC = " + str(accuracy_SVC))
    
        
    file.write("\n" + "====== Results on labeled data (Brexit) ======" + "\n")
    file.write('===== ACCURACY MLP : ' + str(accuracy_MLP) + '=====\n')
    file.write('===== ACCURACY Linear_SVC : ' + str(accuracy_SVC) + '=====\n')
    
    file.close()













