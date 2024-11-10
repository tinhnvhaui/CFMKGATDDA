import random
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
A_LD=np.array(pd.read_csv("05-lncRNA-disease.csv", header=None))
#A_MD=np.array([[1,0],[1,0],[0,1],[0,1],[1,1]])
print(A_LD.shape)
print(A_LD)
# Tinh lai ma tran lncRNA moi
chiso1=0
chiso2=0
dem=0
dataframe1=[]
#dataframe2=pd.DataFrame()
dayso=[]
#vitri=pd.DataFrame()
for i in range(0, A_LD.shape[0]):
    for j in range (0, A_LD.shape[1]):
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

A_testnew_5fold_fold01=A_LD
for k1 in random.sample(range(98880), 19776):
    chisohang=matranchiso[k1, 0]
    chisocot=matranchiso[k1, 1]
    A_testnew_5fold_fold01[chisohang, chisocot]=0

print('A test new la', A_testnew_5fold_fold01)
np.savetxt('A_LD_5fold_fold01_29032024.txt', A_testnew_5fold_fold01.astype(float), fmt='%.8f')
A_LD=A_testnew_5fold_fold01
A_LM=np.array(pd.read_csv("06-lncRNA-miRNA.csv", header=None))
#A_LM=np.array([[1,1],[1,0],[0,1],[0,0],[0,0]])
print(A_LM.shape)
print(A_LM)
#A_LD=A_DL.T
A_LDM=np.concatenate((A_LD,A_LM), axis=1)
print(A_LDM.shape)
print(A_LDM)
num_lncrna=A_LDM.shape[0]
num_disease=A_LD.shape[1]
num_mirna=A_LM.shape[1]
num_sumDM=A_LDM.shape[1]

R_nn=np.zeros((num_lncrna, num_lncrna))
print(R_nn.shape)
for i in range(0, num_lncrna):
    for j in range(0, num_lncrna):
        if (i==j):
            R_nn[i,j]=1
        else:
            count_common_neibought=0
            for k in range(0,num_sumDM):
                if (A_LDM[i,k]==A_LDM[j,k]==1):
                    count_common_neibought+=1
            if (count_common_neibought>0):
                R_nn[i, j] = 1
print(R_nn)

for ic in range(0, num_lncrna):
    for jc in range(0, num_lncrna):
        if (ic == jc):
            R_nn[ic,jc] = 0
        else:
            count_common_neibought_c=0
            for kc in range(0,num_sumDM):
                if (A_LDM[ic,kc]==A_LDM[jc,kc]==1):
                    count_common_neibought_c+=1
            intersection=count_common_neibought_c
            sum1=A_LDM.sum(axis=1)[ic]
            sum2=A_LDM.sum(axis=1)[jc]
            if ((sum1==0) or (sum2==0)):
                R_nn[ic, jc] = 0
            else:
                R_nn[ic, jc]=intersection/math.sqrt(sum1*sum2)

print(R_nn)
R1_LDM=R_nn.dot(A_LDM)
print(R1_LDM)
recommend_ALDM=np.zeros((num_lncrna,num_sumDM))

for i in range(0, num_lncrna):
    for j in range(0,num_sumDM):
        if (A_LDM[i,j]==1):
            recommend_ALDM[i,j]=1

for j_column in range(0,num_sumDM):
    sum_column=0.0
    count_column=A_LDM.sum(axis=0)[j_column]

    for i_row in range(0, num_lncrna):
        if (A_LDM[i_row ,j_column]==1):
            sum_column+=float(R1_LDM[i_row ,j_column])
    p_value_column=sum_column/count_column

    for i_row in range(0, num_lncrna):
        if (A_LDM[i_row ,j_column]==0):
            if (R1_LDM[i_row ,j_column]>p_value_column):
                recommend_ALDM[i_row ,j_column]=1

print('ma tran moi la:')
print(recommend_ALDM)

A_LD_new=np.zeros((num_lncrna, num_disease))
A_LM_new=np.zeros((num_lncrna, num_mirna))
for i_split in range(0, num_lncrna):
    for j_split in range(0, num_disease):
        A_LD_new[i_split, j_split]=recommend_ALDM[i_split, j_split]
    for j_split_mirna in range(num_lncrna, num_sumDM):
        A_LM_new[i_split, j_split_mirna-num_disease]=recommend_ALDM[i_split, j_split_mirna]

print('Kich thuoc A_LD_new_la',A_LD_new.shape)
print('A_LD_new la',A_LD_new)
print('Kich thuoc A_LM_new_la',A_LM_new.shape)
print('A_LM_new la', A_LM_new)
np.savetxt('New_lncrna_dissease_matrix_5fold_fold01_29032024.txt', A_LD_new.astype(int))
np.savetxt('New_lncrna_Micrna_matrix_5fold_fold01_29032024.txt', A_LM_new.astype(int))



