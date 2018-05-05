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
    #                               prod x term                               #
    ###########################################################################
    
    nb_composante = 300
    
    
                #############    Sans TF-IDF    ################
    
    csr_meta = load_npz("matrices/prod_term_matrix.npz")
    clf_pt = TruncatedSVD(nb_composante)
    svd_pt = clf_pt.fit_transform(csr_meta)
    variance_pt = sorted(clf_pt.explained_variance_ratio_, reverse=True) #[0.278, 0.0328, 0.0180, ...]
    print(variance_pt[0:20])
    
    #plots inertie :
#    plt.figure(figsize=(15,9))
#    #plt.ylim((0 , 1))
#    indices = [x+1 for x in range(len(variance_pt))]
#    plt.bar(indices , variance_pt)
#    plt.xticks([1] + list(range(10, len(variance_pt)+1 , 10)))
#    plt.xlabel('axe')
#    plt.ylabel('inertie')
#    plt.title('Produit x terme : Variance expliquée par les premiers axes de la SVD')
#    plt.savefig('inertie_svd_prod_term' + str(nb_composante) + '.png')
    
    #cumulative variance
    cumul = [variance_pt[0]]
    for i in range( 1 , len(variance_pt)):
        cumul.append(cumul[i-1] + variance_pt[i])
        
#    plt.figure(figsize=(8,6))
#    plt.plot(cumul)
#    plt.xlabel('axe')
#    plt.ylabel('variance')
#    plt.title('Produit x terme : Variance cumulée des axes de la SVD')
#    plt.savefig('var_cumulee_prod_term' + str(nb_composante) + '.png')
#    
#    #plot SVD
#    plot_svd(svd_pt, 1, 2, 'svd_1_2_ProdTerm' + str(nb_composante) + '.png')
#    plot_svd(svd_pt, 2, 3, 'svd_2_3_ProdTerm' + str(nb_composante) + '.png')
        
    file = open("resultat_SVD.txt","a")
    file.write("\n\n\n\n*******************************************************\n")
    file.write("\nproduit terme : \n")
    file.write("Inertie : ")
    file.write(str(variance_pt[0])  + ' - ' +  str(variance_pt[1]))
    file.write("Inertie cumulée : ")
    file.write(str(cumul[nb_composante-1]))
    
    
    
                ##############    Avec TF-IDF    ################
    
    transformer = TfidfTransformer(use_idf = True)
    tf_idf_pt = transformer.fit_transform(csr_meta)
    clf_pt_tf = TruncatedSVD(nb_composante)
    svd_tfidf_pt = clf_pt_tf.fit_transform(tf_idf_pt)
    variance_pt_tf = sorted(clf_pt_tf.explained_variance_ratio_, reverse=True) #[0.0128, 0.0078, 0.0056, ...]
    
    #plots inertie :
#    plt.figure(figsize=(15,9))
#    #plt.ylim((0 , 1))
#    indices = [x+1 for x in range(len(variance_pt_tf))]
#    plt.bar(indices , variance_pt_tf)
#    plt.xticks([1] + list(range(10, len(variance_pt_tf)+1 , 10)))
#    plt.xlabel('axe')
#    plt.ylabel('inertie')
#    plt.title('Produit x terme : Variance expliquée par les premiers axes de la SVD (tf-idf)')
#    plt.savefig('inertie_svd_prod_term_tfidf' + str(nb_composante) + '.png')
    
    
    #cumulative variance
    cumul_tf = [variance_pt_tf[0]]
    for i in range( 1 , len(variance_pt_tf)):
        cumul_tf.append(cumul_tf[i-1] + variance_pt_tf[i])
        
#    plt.figure(figsize=(8,6))
#    plt.plot(cumul_tf)
#    plt.xlabel('axe')
#    plt.ylabel('variance')
#    plt.title('Produit x terme : Variance cumulée des axes de la SVD (TF-IDF)')
#    plt.savefig('var_cumulee_prod_term_tfidf' + str(nb_composante) + '.png')
#        
#    
        
    file.write("\nproduit terme TF-IDF : \n")
    file.write("Inertie : ")
    file.write(str(variance_pt_tf[0])  + ' - ' +  str(variance_pt_tf[1]))
    file.write("Inertie cumulée : ")
    file.write(str(cumul_tf[nb_composante-1]))

    
    file.close()
    
    







