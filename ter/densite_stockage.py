#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:01:32 2018

@author: mira
"""


import numpy as np
from scipy.sparse import load_npz
from os import chdir
chdir("/home/mira/TAF/TER/code")



prod_user = load_npz("matrices/prod_user_matrix.npz")

dense = np.prod(prod_user.toarray().shape) * prod_user.toarray().itemsize / 1e6
sparse = (prod_user.data.nbytes + prod_user.indptr.nbytes + prod_user.indices.nbytes)/1e6

