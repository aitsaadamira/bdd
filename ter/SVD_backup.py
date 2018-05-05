#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:33:22 2018

@author: mira
"""


from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
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
    #                               prod x term                               #
    ###########################################################################
    
    
                ##############    Sans TF-IDF    ################
    
    csr_meta = load_npz("matrices/prod_term_matrix.npz")
    clf_pt = TruncatedSVD(300)
    svd_pt = clf_pt.fit_transform(csr_meta)
    variance_pt = sorted(clf_pt.explained_variance_ratio_, reverse=True) #[0.278, 0.0328, 0.0180, ...]
    
    #plots inertie :
    plt.figure(figsize=(15,9))
    #plt.ylim((0 , 1))
    indices = [x+1 for x in range(len(variance_pt))]
    plt.bar(indices , variance_pt)
    plt.xticks([1] + list(range(10, len(variance_pt)+1 , 10)))
    plt.xlabel('axe')
    plt.ylabel('inertie')
    plt.title('Produit x terme : Variance expliquée par les premiers axes de la SVD')
    plt.savefig('inertie_svd_prod_term.png')
    
    #cumulative variance
    cumul = [variance_pt[0]]
    for i in range( 1 , len(variance_pt)):
        cumul.append(cumul[i-1] + variance_pt[i])
        
    plt.figure(figsize=(8,6))
    plt.plot(cumul)
    plt.xlabel('axe')
    plt.ylabel('variance')
    plt.title('Produit x terme : Variance cumulée des axes de la SVD')
    plt.savefig('var_cumulee_prod_term.png')
    
    #plot SVD
    plot_svd(svd_pt, 1, 2, "svd_1_2_ProdTerm.png")
    plot_svd(svd_pt, 2, 3, "svd_2_3_ProdTerm.png")
    
    
    
                ##############    Avec TF-IDF    ################
    
    transformer = TfidfTransformer(use_idf = True)
    tf_idf_pt = transformer.fit_transform(csr_meta)
    clf_pt_tf = TruncatedSVD(300)
    svd_tfidf_pt = clf_pt_tf.fit_transform(tf_idf_pt)
    variance_pt_tf = sorted(clf_pt_tf.explained_variance_ratio_, reverse=True) #[0.0128, 0.0078, 0.0056, ...]
    
    #plots inertie :
    plt.figure(figsize=(15,9))
    #plt.ylim((0 , 1))
    indices = [x+1 for x in range(len(variance_pt_tf))]
    plt.bar(indices , variance_pt_tf)
    plt.xticks([1] + list(range(10, len(variance_pt_tf)+1 , 10)))
    plt.xlabel('axe')
    plt.ylabel('inertie')
    plt.title('Produit x terme : Variance expliquée par les premiers axes de la SVD (tf-idf)')
    plt.savefig('inertie_svd_prod_term_tfidf.png')
    
    
    #cumulative variance
    cumul_tf = [variance_pt_tf[0]]
    for i in range( 1 , len(variance_pt_tf)):
        cumul_tf.append(cumul_tf[i-1] + variance_pt_tf[i])
        
    plt.figure(figsize=(8,6))
    plt.plot(cumul_tf)
    plt.xlabel('axe')
    plt.ylabel('variance')
    plt.title('Produit x terme : Variance cumulée des axes de la SVD (TF-IDF)')
    plt.savefig('var_cumulee_prod_term_tfidf.png')
        
    
#    ###########################################################################
#    #                               prod x user                               # 
#    ###########################################################################
#    
#
#                ##############    Sans TF-IDF    ################
#
#    
#    csr_rev = load_npz("matrices/prod_user_matrix.npz")
#    
#    clf_pu = TruncatedSVD(300)
#    svd_pu = clf_pu.fit_transform(csr_rev)
#    variance_pu = sorted(clf_pu.explained_variance_ratio_, reverse=True) #[0.0041, 0.0032, 0.0031, ...
#    
#    #plots inertie :
#    plt.figure(figsize=(15,9))
#    #plt.ylim((0 , 1))
#    indices = [x+1 for x in range(len(variance_pu))]
#    plt.bar(indices , variance_pu)
#    plt.xticks([1] + list(range(10, len(variance_pu)+1 , 10)))
#    plt.xlabel('axe')
#    plt.ylabel('inertie')
#    plt.title('Produit x user : Variance expliquée par les premiers axes de la SVD')
#    plt.savefig('inertie_svd_prod_user.png')
#    
#    
#    #cumulative variance
#    cumul = [variance_pu[0]]
#    for i in range( 1 , len(variance_pu)):
#        cumul.append(cumul[i-1] + variance_pu[i])
#        
#    plt.figure(figsize=(8,6))
#    plt.plot(cumul)
#    plt.xlabel('axe')
#    plt.ylabel('variance')
#    plt.title('Produit x user : Variance cumulée des axes de la SVD')
#    plt.savefig('var_cumulee_prod_user.png')
#    
#    
#    #plot SVD
#    plot_svd(svd_pu, 1, 2, "svd_1_2_ProdUser.png")
#    plot_svd(svd_pu, 1, 2, "svd_2_3_ProdUser.png")
#    
#    
#    
#                ##############    Avec TF-IDF    ################
#   
#    transformer = TfidfTransformer(use_idf = True)
#    tf_idf_pu = transformer.fit_transform(csr_rev)
#    clf_pu_tf = TruncatedSVD(300)
#    svd_tfidf_pu = clf_pu_tf.fit_transform(tf_idf_pu)
#    variance_pu_tf = sorted(clf_pu_tf.explained_variance_ratio_, reverse=True) #[0.0128, 0.0078, 0.0056, ...]
#    
#    #plots inertie :
#    plt.figure(figsize=(15,9))
#    #plt.ylim((0 , 1))
#    indices = [x+1 for x in range(len(variance_pu_tf))]
#    plt.bar(indices , variance_pu_tf)
#    plt.xticks([1] + list(range(10, len(variance_pu_tf)+1 , 10)))
#    plt.xlabel('axe')
#    plt.ylabel('inertie')
#    plt.title('Produit x user : Variance expliquée par les premiers axes de la SVD (tf-idf)')
#    plt.savefig('inertie_svd_prod_term_tfidf.png')
#    
#    
#    #cumulative variance
#    cumul_pu_tf = [variance_pu_tf[0]]
#    for i in range( 1 , len(variance_pu_tf)):
#        cumul_pu_tf.append(cumul_pu_tf[i-1] + variance_pu_tf[i])
#        
#    plt.figure(figsize=(8,6))
#    plt.plot(cumul_pu_tf)
#    plt.xlabel('axe')
#    plt.ylabel('variance')
#    plt.title('Produit x user : Produit x user : Variance cumulée des axes de la SVD (TF-IDF)')
#    plt.savefig('var_cumulee_prod_user_tfidf.png')
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#
#
#
#
#
#
#
