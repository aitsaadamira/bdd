#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:07:59 2018

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
    
    nb_composante = 300
    
    ###########################################################################
    #                               prod x user                               # 
    ###########################################################################
    

                #############    Sans TF-IDF    ################

    
    csr_rev = load_npz("matrices/prod_user_matrix.npz")
    
    clf_pu = TruncatedSVD(nb_composante)
    svd_pu = clf_pu.fit_transform(csr_rev)
    variance_pu = sorted(clf_pu.explained_variance_ratio_, reverse=True) #[0.0041, 0.0032, 0.0031, ...
    
    #plots inertie :
#    plt.figure(figsize=(15,9))
#    #plt.ylim((0 , 1))
#    indices = [x+1 for x in range(len(variance_pu))]
#    plt.bar(indices , variance_pu)
#    plt.xticks([1] + list(range(10, len(variance_pu)+1 , 10)))
#    plt.xlabel('axe')
#    plt.ylabel('inertie')
#    plt.title('Produit x user : Variance expliquée par les premiers axes de la SVD')
#    plt.savefig('inertie_svd_prod_user' + str(nb_composante) + '.png')  
    
    #cumulative variance
    cumul = [variance_pu[0]]
    for i in range( 1 , len(variance_pu)):
        cumul.append(cumul[i-1] + variance_pu[i])
        
#    plt.figure(figsize=(8,6))
#    plt.plot(cumul)
#    plt.xlabel('axe')
#    plt.ylabel('variance')
#    plt.title('Produit x user : Variance cumulée des axes de la SVD')
#    plt.savefig('var_cumulee_prod_user' + str(nb_composante) + '.png')
#    
#    
#    #plot SVD
#    plot_svd(svd_pu, 1, 2, 'svd_1_2_ProdUser' + str(nb_composante) + '.png')
#    plot_svd(svd_pu, 1, 2, 'svd_2_3_ProdUser' + str(nb_composante) + '.png')
    
    file = open("resultat_SVD.txt","a")
    file.write("\n\n\n\n*******************************************************\n")
    file.write("\nproduit user : \n")
    file.write("Inertie : ")
    file.write(str(variance_pu[0])  + ' - ' +  str(variance_pu[1]))
    file.write("Inertie cumulée : ")
    file.write(str(cumul[nb_composante-1])    )
    
    
                ##############    Avec TF-IDF    ################
   
    transformer = TfidfTransformer(use_idf = True)
    tf_idf_pu = transformer.fit_transform(csr_rev)
    clf_pu_tf = TruncatedSVD(nb_composante)
    svd_tfidf_pu = clf_pu_tf.fit_transform(tf_idf_pu)
    variance_pu_tf = sorted(clf_pu_tf.explained_variance_ratio_, reverse=True) #[0.0128, 0.0078, 0.0056, ...]
    
    #plots inertie :
#    plt.figure(figsize=(15,9))
#    #plt.ylim((0 , 1))
#    indices = [x+1 for x in range(len(variance_pu_tf))]
#    plt.bar(indices , variance_pu_tf)
#    plt.xticks([1] + list(range(10, len(variance_pu_tf)+1 , 10)))
#    plt.xlabel('axe')
#    plt.ylabel('inertie')
#    plt.title('Produit x user : Variance expliquée par les premiers axes de la SVD (tf-idf)')
#    plt.savefig('inertie_svd_prod_user_tfidf' + str(nb_composante) + '.png')
    
    
    #cumulative variance
    cumul_pu_tf = [variance_pu_tf[0]]
    for i in range( 1 , len(variance_pu_tf)):
        cumul_pu_tf.append(cumul_pu_tf[i-1] + variance_pu_tf[i])
        
#    plt.figure(figsize=(8,6))
#    plt.plot(cumul_pu_tf)
#    plt.xlabel('axe')
#    plt.ylabel('variance')
#    plt.title('Produit x user : Produit x user : Variance cumulée des axes de la SVD (TF-IDF)')
#    plt.savefig('var_cumulee_prod_user_tfidf' + str(nb_composante) + '.png')

    file = open("resultat_SVD.txt","a")
    file.write("n\produit user  TF-IDF : \n")
    file.write("Inertie : ")
    file.write(str(variance_pu_tf[0])  + ' - ' +  str(variance_pu_tf[1]))
    file.write("Inertie cumulée : ")
    file.write(str(cumul_pu_tf[nb_composante-1])    )
    
    
    file.close()
    
    
    
    