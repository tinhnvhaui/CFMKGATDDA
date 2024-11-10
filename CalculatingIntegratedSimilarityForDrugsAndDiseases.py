import csv
import numpy as np
import pandas as pd
import scipy.odr as sci
from sklearn import preprocessing as pp
import random
import math
DiSS=np.array(pd.read_csv("CDataset_DiseaseSim.csv", header=None))
DiGausS=np.array(pd.read_csv("GaussianSimilarityForDiseases_CDataset_141024.csv", header=None))
DiCosineS=np.array(pd.read_csv("CosineSimilarityForDiseases_CDataset141024.csv", header=None))
DiSigmoidS=np.array(pd.read_csv("SigmoidSimilarityForDiseases_CDataset141024.csv", header=None))
DrSS=np.array(pd.read_csv("CDataset_DrugSim.csv", header=None))
DrGausS=np.array(pd.read_csv("GaussianSimilarityForDrugs_CDataset_141024.csv", header=None))
DrCosineS=np.array(pd.read_csv("CosineSimilarityForDrugs_CDataset141024.csv", header=None))
DrSigmoidS=np.array(pd.read_csv("SigmoidSimilarityForDrugs_CDataset141024.csv", header=None))
nDi=DiSS.shape[0]
IDiseaseSimilarity=np.zeros([nDi,nDi])
for i in range(nDi):
    for j in range(nDi):
        if (DiSS[i,j]==1):
            IDiseaseSimilarity[i, j] = DiSS[i,j]
        else:
            IDiseaseSimilarity[i, j]= (DiSS[i,j]+DiGausS[i,j]+ DiCosineS[i,j]+ DiSigmoidS[i,j])/4

nDr=DrSS.shape[0]
IDrugSimilarity=np.zeros([nDr, nDr])
for i in range(nDr):
    for j in range(nDr):
        if (DrSS[i,j]==1):
            IDrugSimilarity[i, j] = DrSS[i,j]
        else:
            IDrugSimilarity[i, j]= (DrSS[i,j]+DrGausS[i,j]+ DrCosineS[i,j]+ DrSigmoidS[i,j])/4

print(IDiseaseSimilarity.shape)
print(IDiseaseSimilarity)
print(IDrugSimilarity.shape)
print(IDrugSimilarity)
np.savetxt('IntegratedSimilarityForDiseases_CDataset14102024New.txt', IDiseaseSimilarity.astype(float))
np.savetxt('IntegratedSimilarityForDrugs_CDataset14102024New.txt', IDrugSimilarity.astype(float))
