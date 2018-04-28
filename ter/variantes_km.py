#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:19:19 2018

@author: mira
"""

import numpy as np
import pandas as pd
from math import sqrt
from pickle import dump
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
from scipy.sparse import load_npz, csr_matrix
from coclust.evaluation.external import accuracy
from sklearn.preprocessing import Normalizer, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import normalized_mutual_info_score as RMI, adjusted_rand_score as Rand


def save_res(res, fun_name, mat_name, nb_clusters, nb_init):
    filename = "result/" +  fun_name + "_" + mat_name + ".pkl"
    with open(filename, 'wb') as file:  
        dump(res, file)


def get_mat(clustering_list, name_list, fun):
    mat = pd.DataFrame(index = name_list, columns = name_list)
    
    for c1 in range(0 , len(clustering_list)):
        for c2 in range(0 , len(clustering_list)):
            mat.set_value(name_list[c1], name_list[c2] , fun(clustering_list[c1].labels_, clustering_list[c2].labels_))

    return mat


def compare_clustering(clustering_list, name_list, fun_name, mat_name):
    """
        Param : list of clustering results
        Saves RMI, Rand and Accuracy matrices as CSV files
    """

    mat_RMI = get_mat(clustering_list, name_list, RMI)
    mat_Rand = get_mat(clustering_list, name_list, Rand)
    mat_acc = get_mat(clustering_list, name_list, accuracy)
    
    file_prefix = "result/"
    file_suffix =  fun_name + "_" + mat_name
    
    mat_RMI.to_csv(file_prefix + "RMI_" + file_suffix + ".csv")
    mat_Rand.to_csv(file_prefix + "Rand_" + file_suffix + ".csv")
    mat_acc.to_csv(file_prefix + "Acc_" + file_suffix + ".csv")
    

def kmeans(matrix, n_clusters, nb_init):
    labeler = KMeans(n_clusters = n_clusters, n_init=nb_init, max_iter=100)
    return labeler.fit(matrix)


def sphe_kmeans(matrix, n_clusters, nb_init):
    labeler = SphericalKMeans(n_clusters = n_clusters, n_init=nb_init,  max_iter=100)
    return labeler.fit(matrix)
    

def tfidf(csr):
    transformer = TfidfTransformer(use_idf = True)
    return transformer.fit_transform(csr)


def chi2(matrix):
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    sum_rows = matrix.sum(1)
    sum_cols = matrix.sum(0)
    
    data_new = []
    
    for i in range(0, nrows):
        indices = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
        data = matrix.data[matrix.indptr[i]:matrix.indptr[i+1]]
    
        for j in range(0, len(indices)):
            data_new.append(data[j] / sqrt(sum_rows[i,0]*sum_cols[0,indices[j]]))
        
    normalized = csr_matrix( (np.array(data_new), matrix.indices, matrix.indptr), shape=(nrows,ncols))
    return normalized

def kmeans_tf_idf(matrix, nb_clusters, nb_init, fun):
    matrix_tfidf = tfidf(matrix)
    km = fun(matrix_tfidf, nb_clusters, nb_init=nb_init)
    return km


def kmeans_norm_line(matrix, nb_clusters, nb_init, fun):
    matrix_norm = normalize(matrix)
    km = fun(matrix_norm, nb_clusters, nb_init=nb_init)
    return km


def kmeans_norm_unit(matrix, nb_clusters, nb_init, fun):
    matrix_norm = Normalizer().fit_transform(matrix)
    km = fun(matrix_norm, nb_clusters, nb_init=nb_init)
    return km


def kmeans_chi2(matrix, nb_clusters, nb_init, fun):
    matrix_tfidf = chi2(matrix)
    km = fun(matrix_tfidf, nb_clusters, nb_init=nb_init)
    return km
    

def kmeans_exec(matrix, nb_clusters, nb_init, fun, fun_name, mat_name):
    """
        Applies k-means or spherical k-means, depending on the "fun" attribute. 5 variants :
            - with raw matrix
            - with tf-idf normalization
            - with raw scaling to unit form
            - with individual cell scaling to unit form
            - with chi2 normalization
            
    """

    km_nothing = (matrix, nb_clusters, nb_init)
    save_res(km_nothing, "k-means tfidf" , "prod_term" , nb_clusters, nb_init)
    
    km_tfidf = kmeans_tf_idf(matrix, nb_clusters, nb_init, fun)
    save_res(km_tfidf, "k-means tfidf" , "prod_term" , nb_clusters, nb_init)
    
    km_norm_line = kmeans_norm_line(matrix, nb_clusters, nb_init, fun)
    save_res(km_norm_line, "k-means norm_line" , "prod_term" , nb_clusters, nb_init)
    
    km_norm_unit = kmeans_norm_unit(matrix, nb_clusters, nb_init, fun)
    save_res(km_norm_unit, "k-means km_norm_unit" , "prod_term" , nb_clusters, nb_init)
    
    km_chi2 = kmeans_chi2(matrix, nb_clusters, nb_init, fun)
    save_res(km_chi2, "k-means km_chi2" , "prod_term" , nb_clusters, nb_init)
    
    clustering_list = [km_tfidf, km_norm_line, km_norm_unit, km_chi2]
    name_list = ["tfidf" , "notm_line" , "notm_unit" , "chi2"]
    
    compare_clustering(clustering_list, name_list = name_list, fun_name = fun_name, mat_name = mat_name)
    

if __name__ == "__main__":    
    
    nb_clusters = 200
    nb_init = 5
    
    ###########################################################################
    #                             PROD x TERM                                 #
    
    prod_term = load_npz("prod_term_matrix.npz")

    kmeans_exec(matrix = prod_term, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "prod_term")
    kmeans_exec(matrix = prod_term, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "prod_term")
    
    ###########################################################################
    #                             PROD x USER                                 #
    
    prod_user = load_npz("prod_user_matrix.npz")

    kmeans_exec(matrix = prod_user, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "prod_user")
    kmeans_exec(matrix = prod_user, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "prod_user")
    
    ###########################################################################
    #                          PROD x PROD (sub)                              #
    
    csr_sub = load_npz("csr_sub.npz")

    kmeans_exec(matrix = csr_sub, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "sub_mat")
    kmeans_exec(matrix = csr_sub, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "sub_mat")
    
    ###########################################################################
    #                          PROD x PROD (comp)                             #
    
    csr_comp = load_npz("csr_comp.npz")

    kmeans_exec(matrix = csr_comp, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "sub_comp")
    kmeans_exec(matrix = csr_comp, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "sub_comp")









