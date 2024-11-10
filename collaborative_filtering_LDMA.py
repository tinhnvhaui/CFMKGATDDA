import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
A_DrDi=np.array(pd.read_csv("ADrugDisease_5fold_fold04_23102024.csv", header=None))
print(A_DrDi.shape)
print(A_DrDi)
num_Drugs=A_DrDi.shape[0]
num_Dis=A_DrDi.shape[1]

R_nn=np.zeros((num_Drugs, num_Drugs))
print(R_nn.shape)
for i in range(num_Drugs):
    for j in range(num_Drugs):
        if i == j:
            R_nn[i, j] = 1
        elif np.sum(A_DrDi[i, :] == A_DrDi[j, :]) > 0:
            R_nn[i, j] = 1

print(R_nn)

for ic in range(0, num_Drugs):
    for jc in range(0, num_Drugs):
        if (ic == jc):
            R_nn[ic,jc] = 0
        else:
            intersection = len(np.intersect1d(np.where(A_DrDi[ic, :]), np.where(A_DrDi[jc, :])))
            product = np.sum(A_DrDi[ic, :]) * np.sum(A_DrDi[jc, :])
            if (product==0):
                R_nn[ic, jc]=0
            else:
                R_nn[ic,jc] = intersection / np.sqrt(product)

print(R_nn)
R1_A_DrDi=R_nn.dot(A_DrDi)
print(R1_A_DrDi)
recommend_A_DrDi=np.zeros((num_Drugs,num_Dis))

for i in range(0, num_Drugs):
    for j in range(0,num_Dis):
        if (A_DrDi[i,j]==1):
            recommend_A_DrDi[i,j]=1
    for i in range(num_Dis):
        index1 = np.where(A_DrDi[:, i])[0]
        temp = np.sum(R1_A_DrDi[index1, i])
        index2 = np.where(A_DrDi[:, i] == 0)[0]
        if (len(index1)==0):
            recommend_A_DrDi[index2[j], i] = 0
        else:
            ave = np.sum(R1_A_DrDi[index1, i]) / len(index1)

            for j in range(len(index2)):
                if R1_A_DrDi[index2[j], i] > ave:
                    recommend_A_DrDi[index2[j], i] = 1


print('ma tran moi la:')
print(recommend_A_DrDi)

A_DrDi_new=recommend_A_DrDi
#A_LD_new=A_LD_new.T
print('Kich thuoc A_DrDi_new',A_DrDi_new.shape)
print('A_DrDi_new',A_DrDi_new)

np.savetxt('C_Dataset_CF_5fold_fold04_UpMatrix23102024.txt', A_DrDi_new.astype(int))




