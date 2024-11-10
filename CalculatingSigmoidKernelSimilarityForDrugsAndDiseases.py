import csv
import numpy as np
import pandas as pd
import scipy.odr as sci
from sklearn import preprocessing as pp
import random
import math

def sig_kr(DiDr):
    nDi = DiDr.shape[0]
    nDr = DiDr.shape[1]
    sig_MS1 = []
    sig_DS1 = []
    for i in range(nDi):
        for j in range(nDi):
            a = DiDr[i, :]
            b = DiDr[j, :]
            z = (1 / nDi) * (np.dot(a, b))
            sig_ms = math.tanh(z)
            sig_MS1.append(sig_ms)

    for i in range(nDr):
        for j in range(nDr):
            a1 = DiDr[:, i]
            b1 = DiDr[:, j]
            z1 = (1 / nDr) * (np.dot(a1, b1))
            sig_ds = math.tanh(z1)
            sig_DS1.append(sig_ds)

    sig_MS1 = np.array(sig_MS1).reshape(nDi, nDi)
    sig_DS1 = np.array(sig_DS1).reshape(nDr, nDr)
    return sig_MS1, sig_DS1

DiDr=np.array(pd.read_csv("CDataset_DiDrA.csv", header=None))
print(' kich thuoc ma tran la 1', DiDr.shape)
print('known_interaction=', DiDr)
nDi=DiDr.shape[0]
nDr=DiDr.shape[1]
print('nDi=', nDi)
print('nDr=', nDr)
SigDi, SigDr=sig_kr(DiDr)
print('kd=', SigDi)
print('km=', SigDr)
np.savetxt('SigmoidSimilarityForDiseases_CDataset141024.txt', SigDi.astype(float))
np.savetxt('SigmoidSimilarityForDrugs_CDataset141024.txt', SigDr.astype(float))