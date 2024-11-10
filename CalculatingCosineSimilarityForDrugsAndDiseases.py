import csv
import numpy as np
import pandas as pd
import scipy.odr as sci
from sklearn import preprocessing as pp
import random
import math

#from numpy.linalg import norm as normal
def cos_sim(DiDr):
    nDi = DiDr.shape[0]
    nDr = DiDr.shape[1]
    cos_MS1 = []
    cos_DS1 = []
    for i in range(nDi):
        for j in range(nDi):
            a = DiDr[i, :]
            b = DiDr[j, :]
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm != 0 and b_norm != 0:
                cos_ms = np.dot(a, b) / (a_norm * b_norm)
                cos_MS1.append(cos_ms)
            else:
                cos_MS1.append(0)

    for i in range(nDr):
        for j in range(nDr):
            a1 = DiDr[:, i]
            b1 = DiDr[:, j]
            a1_norm = np.linalg.norm(a1)
            b1_norm = np.linalg.norm(b1)
            if a1_norm != 0 and b1_norm != 0:
                cos_ds = np.dot(a1, b1) / (a1_norm * b1_norm)
                cos_DS1.append(cos_ds)
            else:
                cos_DS1.append(0)

    cos_MS1 = np.array(cos_MS1).reshape(nDi, nDi)
    cos_DS1 = np.array(cos_DS1).reshape(nDr, nDr)
    return cos_MS1, cos_DS1

DiDr=np.array(pd.read_csv("CDataset_DiDrA.csv", header=None))
print(' kich thuoc ma tran la 1', DiDr.shape)
print('known_interaction=', DiDr)
nDi=DiDr.shape[0]
nDr=DiDr.shape[1]
print('nDi=', nDi)
print('nDr=', nDr)
CosineDi, CosineDr=cos_sim(DiDr)
print('kd=', CosineDi)
print('km=', CosineDr)
np.savetxt('CosineSimilarityForDiseases_CDataset141024.txt', CosineDi.astype(float))
np.savetxt('CosineSimilarityForDrugs_CDataset141024.txt', CosineDr.astype(float))