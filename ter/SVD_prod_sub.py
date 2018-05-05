#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:33:22 2018

@author: mira
"""


from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_svd (svd, axe1, axe2, filename):
    
    plt.figure(figsize=(8,6))
    plt.xlabel('axe 1')
    plt.ylabel('axe 2')
    plt.title('plan ' +str(axe1-1) + '-' + str(axe2-1) + ' de la SVD')
    plt.scatter(svd[:, axe1-1], svd[:, axe2-1], s=3)
    plt.savefig(filename)
    



if __name__ == "__main__":
    
    ###########################################################################
    #                               prod x sub                               #
    ###########################################################################
    
    nb_composante = 300
    
                ##############    Sans TF-IDF    ################
    
    csr_meta = load_npz("/home/mira/TAF/TER/code/matrices/csr_sub.npz")
    clf_sub = TruncatedSVD(nb_composante)
    svd_sub = clf_sub.fit_transform(csr_meta)
    variance_sub = sorted(clf_sub.explained_variance_ratio_, reverse=True) #[0.278, 0.0328, 0.0180, ...]
    
    #plots inertie :
    plt.figure(figsize=(15,9))
    #plt.ylim((0 , 1))
    indices = [x+1 for x in range(len(variance_sub))]
    plt.bar(indices , variance_sub)
    plt.xticks([1] + list(range(10, len(variance_sub)+1 , 10)))
    plt.xlabel('axe')
    plt.ylabel('inertie')
    plt.title('Produit sub : Variance expliquée par les premiers axes de la SVD')
    plt.savefig('inertie_svd_prod_sub' + str(nb_composante) + '.png')
    
    #cumulative variance
    cumul = [variance_sub[0]]
    for i in range( 1 , len(variance_sub)):
        cumul.append(cumul[i-1] + variance_sub[i])
        
    plt.figure(figsize=(8,6))
    plt.plot(cumul)
    plt.xlabel('axe')
    plt.ylabel('variance')
    plt.title('Produit sub : Variance cumulée des axes de la SVD')
    plt.savefig('var_cumulee_prod_sub.png')
    
    #plot SVD
    plot_svd(svd_sub, 1, 2, 'svd_1_2_Prodsub' + str(nb_composante) + '.png')
    plot_svd(svd_sub, 2, 3, 'svd_2_3_Prodsub' + str(nb_composante) + '.png')
    
    
    
                ##############    Avec TF-IDF    ################
    
    transformer = TfidfTransformer(use_idf = True)
    tf_idf_sub = transformer.fit_transform(csr_meta)
    clf_sub_tf = TruncatedSVD(nb_composante)
    svd_tfidf_sub = clf_sub_tf.fit_transform(tf_idf_sub)
    variance_sub_tf = sorted(clf_sub_tf.explained_variance_ratio_, reverse=True) #[0.0128, 0.0078, 0.0056, ...]
    
    #plots inertie :
    plt.figure(figsize=(15,9))
    #plt.ylim((0 , 1))
    indices = [x+1 for x in range(len(variance_sub_tf))]
    plt.bar(indices , variance_sub_tf)
    plt.xticks([1] + list(range(10, len(variance_sub_tf)+1 , 10)))
    plt.xlabel('axe')
    plt.ylabel('inertie')
    plt.title('Produit sub : Variance expliquée par les premiers axes de la SVD (tf-idf)')
    plt.savefig('inertie_svd_prod_sub_tfidf' + str(nb_composante) + '.png')
    
    
    #cumulative variance
    cumul_tf = [variance_sub_tf[0]]
    for i in range( 1 , len(variance_sub_tf)):
        cumul_tf.append(cumul_tf[i-1] + variance_sub_tf[i])
        
    plt.figure(figsize=(8,6))
    plt.plot(cumul_tf)
    plt.xlabel('axe')
    plt.ylabel('variance')
    plt.title('Produit sub : Variance cumulée des axes de la SVD (TF-IDF)')
    plt.savefig('var_cumulee_prod_sub_tfidf' + str(nb_composante) + '.png')
        
    
