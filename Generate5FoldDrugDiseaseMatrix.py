import random
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
A_DDR=np.array(pd.read_csv("CDataset_DiDrA.csv", header=None))
#A_MD=np.array([[1,0],[1,0],[0,1],[0,1],[1,1]])
print(A_DDR.shape)
print(A_DDR)
# Tinh lai ma tran lncRNA moi
chiso1=0
chiso2=0
dem=0
dataframe1=[]
#dataframe2=pd.DataFrame()
dayso=[]
#vitri=pd.DataFrame()
for i in range(0, A_DDR.shape[0]):
    for j in range (0, A_DDR.shape[1]):
        #if (A[i,j]==1):
        dem=dem+1
        chiso1=i
        chiso2=j
        dayso.append(dem)
        dataframe1.append([chiso1, chiso2])

print('day so la:', dayso)
print('so phan tu cua day so', len(dayso))
matranchiso= np.array(dataframe1)
print('matran chi so i la',matranchiso)
print('Kich thuoc ma tran chi so', matranchiso.shape)

A_testnew_5fold_fold01=A_DDR
for k1 in random.sample(range(271167), 54233):
    chisohang=matranchiso[k1, 0]
    chisocot=matranchiso[k1, 1]
    A_testnew_5fold_fold01[chisohang, chisocot]=0

print('A test new la', A_testnew_5fold_fold01)
np.savetxt('ADrugDisease_5fold_fold03_23102024.txt', A_testnew_5fold_fold01.astype(int))
