#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:20:26 2018

@author: mira
"""

from scipy.sparse import load_npz

from sklearn.cluster import KMeans
from pickle import dump

def kmeans(matrix, n_clusters, nb_init):
    labeler = KMeans(n_clusters = n_clusters, n_init=nb_init, max_iter=100)
    return labeler.fit(matrix)



if __name__ == "__main__":
    
    prod_user = load_npz("prod_user_matrix.npz") 
    
    #########################
    KM = kmeans(prod_user, 200, 5)   
    
    filename = "KMEANS_PROD_USER.pkl"
    with open(filename, 'wb') as file:  
        dump(KM, file)
    #########################
    