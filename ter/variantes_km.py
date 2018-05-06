#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 03:18:02 2018

@author: mira
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:19:19 2018

@author: mira
"""

import numpy as np
import pandas as pd
from math import sqrt
from pickle import dump, load
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
from scipy.sparse import load_npz, csr_matrix
from coclust.evaluation.external import accuracy
from sklearn.preprocessing import Normalizer, normalize
from sklearn.feature_extraction.text import TfidfTransformer#, CountVectorizer
from sklearn.metrics import normalized_mutual_info_score as RMI, adjusted_rand_score as Rand
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD


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
    
    mat_RMI.to_csv(file_prefix + "RMI_" + file_suffix + ".csv" , float_format='%.3f')
    mat_Rand.to_csv(file_prefix + "Rand_" + file_suffix + ".csv", float_format='%.3f')
    mat_acc.to_csv(file_prefix + "Acc_" + file_suffix + ".csv", float_format='%.3f')
    

def kmeans(matrix, n_clusters, nb_init):
    labeler = KMeans(n_clusters = n_clusters, n_init=nb_init, max_iter=100)
    print("kmeans")
    return labeler.fit(matrix)


def sphe_kmeans(matrix, n_clusters, nb_init):
    labeler = SphericalKMeans(n_clusters = n_clusters, n_init=nb_init,  max_iter=100)
    print("sphe_kmeans")
    return labeler.fit(matrix)
    

def tfidf(csr):
    transformer = TfidfTransformer(use_idf = True)
    tfidf = transformer.fit_transform(csr)
    print("tf_idf")
    return tfidf

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
    print("chi2")
    return normalized


def LSA(matrix):

    tfidf = TfidfTransformer()
    svd = TruncatedSVD(300)
    normalizer = Normalizer()
    lsa = make_pipeline(tfidf, svd, normalizer)

    res_LSA = lsa.fit_transform(matrix)
    print("LSA")
    return res_LSA


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

def kmeans_lsa(matrix, nb_clusters, nb_init, fun):
    matrix_lsa = LSA(matrix)
    km = fun(matrix_lsa,  nb_clusters, nb_init=nb_init)
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
    
    clustering_list = []
    name_list = []
    
#    km_nothing = load( open( "result/" +  fun_name + "_nothing" + "_" + mat_name + ".pkl" , "rb"))
#
#    print("================== nothing ======================")
#    
#    km_tfidf = load( open( "result/" +  fun_name + "_tfidf" + "_" + mat_name + ".pkl" , "rb"))
#    
#    print("=================== tfidf =======================")
#    
#    km_norm_line = load( open( "result/" +  fun_name + "_norm_line" + "_" + mat_name + ".pkl" , "rb"))
#    
#    print("================== norm_line =====================")
#    
#    km_norm_unit = load( open( "result/" +  fun_name + "_norm_unit" + "_" + mat_name + ".pkl" , "rb"))
#    
#    print("================== norm_unit =====================")
#    
#    km_chi2 = load( open( "result/" +  fun_name + "_chi2" + "_" + mat_name + ".pkl" , "rb"))
#    
#    print("===================== chi2 =======================")
#        
#    km_lsa = load( open( "/home/mira/Downloads/resultats_km/" +  fun_name + "_lsa" + "_" + mat_name + ".pkl" , "rb"))
#    
#    print("====================== lsa =======================")
#    
#    clustering_list = [km_nothing, km_tfidf, km_norm_line, km_norm_unit, km_chi2]
#    name_list = ["/" , "tfidf" , "norm_line" , "norm_unit" , "chi2"]
#    
#    clustering_list.append(km_lsa)
#    name_list.append("lsa")
    
    km_nothing = fun(matrix, nb_clusters, nb_init)
#    save_res(km_nothing, fun_name + "_nothing" , mat_name, nb_clusters, nb_init)
    
    print("================== nothing ======================")
    
    km_tfidf = kmeans_tf_idf(matrix, nb_clusters, nb_init, fun)
#    save_res(km_tfidf, fun_name + "_tfidf" , mat_name , nb_clusters, nb_init)
    
    print("=================== tfidf =======================")
    
    km_norm_line = kmeans_norm_line(matrix, nb_clusters, nb_init, fun)
#    save_res(km_norm_line, fun_name + "_norm_line" , mat_name , nb_clusters, nb_init)
    
    print("================== norm_line =====================")
    
    km_norm_unit = kmeans_norm_unit(matrix, nb_clusters, nb_init, fun)
#    save_res(km_norm_unit, fun_name + "_norm_unit" , mat_name , nb_clusters, nb_init)
    
    print("================== norm_unit =====================")
    
    km_chi2 = kmeans_chi2(matrix, nb_clusters, nb_init, fun)
    save_res(km_chi2, fun_name + "_chi2" , mat_name , nb_clusters, nb_init)
    
    print("===================== chi2 =======================")
        
    clustering_list = [km_nothing, km_tfidf, km_norm_line, km_norm_unit, km_chi2]
    name_list = ["/" , "tfidf" , "norm_line" , "norm_unit" , "chi2"]
    
    km_lsa = kmeans_lsa(matrix, nb_clusters, nb_init, fun)
    save_res(km_lsa, fun_name + "_lsa" , mat_name , nb_clusters, nb_init)
    clustering_list.append(km_lsa)
    name_list.append("lsa")
    
    print("====================== LSA =======================")

    compare_clustering(clustering_list, name_list = name_list, fun_name = fun_name, mat_name = mat_name)
    
    

if __name__ == "__main__":    
    
    nb_clusters = 200
    nb_init = 5
    
#    print("prod_term")
    
    ###########################################################################
#    #                             PROD x TERM                                 #
#    
#    prod_term = load_npz("matrices/prod_term_matrix.npz")
##
#    kmeans_exec(matrix = prod_term, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "prod_term")
#    kmeans_exec(matrix = prod_term, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "prod_term")
##    
#    
    print("prod_user")
    
    ###########################################################################
    #                             PROD x USER                                 #
    
    prod_user = load_npz("matrices/prod_user_matrix.npz")

    kmeans_exec(matrix = prod_user, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "prod_user")
#    kmeans_exec(matrix = prod_user, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "prod_user")
    
#    print("prod_sub")
#    
#    ###########################################################################
#    #                          PROD x PROD (sub)                              #
#    
#    csr_sub = load_npz("matrices/csr_sub.npz")
#
#    kmeans_exec(matrix = csr_sub, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "sub_mat")
#    kmeans_exec(matrix = csr_sub, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "sub_mat")
#    
#    print("prod_comp")
#    
#    ###########################################################################
#    #                          PROD x PROD (comp)                             #
#    
#    csr_comp = load_npz("matrices/csr_comp.npz")
#
#    kmeans_exec(matrix = csr_comp, nb_clusters = nb_clusters, nb_init = nb_init, fun = kmeans, fun_name = "kmeans" , mat_name = "comp_mat")
#    kmeans_exec(matrix = csr_comp, nb_clusters = nb_clusters, nb_init = nb_init, fun = sphe_kmeans, fun_name = "sphe_kmeans" , mat_name = "comp_mat")









