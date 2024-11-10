import csv
import numpy as np
import pandas as pd
import scipy.odr as sci
from sklearn import preprocessing as pp
import random
import math

#from numpy.linalg import norm as normal

def calculateGaussianSimilarity(known_interaction, nD, nM):
    # calculate gamad for Gaussian kernel calculation
    C = known_interaction
    nD = C.shape[0]
    nM = C.shape[1]
    gamaD=nD/(np.linalg.norm(known_interaction,'fro')**2)
    kd=np.zeros([nD,nD])
    D=C.dot(C.T)
    for i in range(0, nD):
        for j in range(i, nD):
            kd[i,j]= math.exp(-gamaD*(D[i,i]+D[j,j]-2*D[i,j]))

    kd=kd+kd.T-np.diag(np.diag(kd))
    gamaM = nM / (np.linalg.norm(known_interaction,'fro') ** 2)
    #gamaM =1
    km=np.zeros([nM, nM])
    E=C.T.dot(C)
    for i in range(0, nM):
        for j in range(i, nM):
            km[i, j] = math.exp(-gamaM * (E[i, i] + E[j, j] - 2 * E[i, j]))

    km = km + km.T - np.diag(np.diag(km))
    return kd, km

known_interaction=np.array(pd.read_csv("CDataset_DiDrA.csv", header=None))
print(' kich thuoc ma tran la 1', known_interaction.shape)
print('known_interaction=', known_interaction)
nD=known_interaction.shape[0]
nM=known_interaction.shape[1]
print('nD=', nD)
print('nM=', nM)
kd, km=calculateGaussianSimilarity(known_interaction, nD, nM)
print('kd=', kd)
print('km=', km)
np.savetxt('GaussianSimilarityForDiseases_CDataset_141024.txt', kd.astype(float))
np.savetxt('GaussianSimilarityForDrugs_CDataset_141024.txt', km.astype(float))