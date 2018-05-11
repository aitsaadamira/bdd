#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:50:52 2018

@author: mira
"""


from scipy import sparse
from coclust.coclustering import CoclustInfo
from coclust.visualization import plot_delta_kl
from pickle import dump
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from os import chdir
chdir("/home/mira/TAF/TER/code")

def coclustering(X, n_row_clusters, n_col_clusters, nb_ex):
    """
        Co-clustering method on a sparse_matrix
    """
    model = CoclustInfo(n_row_clusters, n_col_clusters, n_init=nb_ex)
    return model.fit(X)

print("prod x term")

mat=sparse.load_npz("matrices/prod_term_matrix.npz")
coclust_prodterm = coclustering(mat, 200, 200, 5)

with open("co_clust_prod_term.pkl", 'wb') as file:  
    dump(coclust_prodterm, file)


print("prod x user")

mat=sparse.load_npz("matrices/prod_user_matrix.npz")
coclust_produser = coclustering(mat, 200, 200, 5)

with open("co_clust_prod_user.pkl", 'wb') as file:  
    dump(coclust_produser, file)

print("sub")

mat=sparse.load_npz("matrices/csr_sub.npz")
coclust_sub = coclustering(mat, 200, 200, 5)

with open("co_clust_sub.pkl", 'wb') as file:  
    dump(coclust_sub, file)


print("comp")


mat=sparse.load_npz("matrices/csr_comp.npz")
coclust_comp = coclustering(mat, 200, 200, 5)

with open("co_clust_comp.pkl", 'wb') as file:  
    dump(coclust_comp, file)



