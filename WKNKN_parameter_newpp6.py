import csv
import numpy as np
import pandas as pd
import scipy.odr as sci
from sklearn import preprocessing as pp
import random
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

def KNN(network_similarity, K):
    nNetRows=network_similarity.shape[0]
    nNetCols=network_similarity.shape[1]
    network_similarity=network_similarity-np.diag(np.diag(network_similarity))
    knn_network=np.zeros([nNetRows, nNetCols])
    sort_network=-np.sort(-network_similarity, axis=1)
    idx=np.argsort(-network_similarity, axis=1)
    for i in range(0, nNetRows):
        for j in range(0,K):
            knn_network[i,idx[i,j]]=sort_network[i,j]

    return knn_network, sort_network, idx

def CalWeightedKNN(known_interaction, diseaseSM, miRNASimilarity, K, r):
    nRows=known_interaction.shape[0]
    nCols=known_interaction.shape[1]
    Y_m=np.zeros([nRows, nCols])
    Y_d=np.zeros([nRows, nCols])
    knn_network_m, sort_network_m, idx_network_m =KNN(miRNASimilarity,K) # for miRNAs

    for i in range(0, nCols):
        w=np.zeros([1, K])
        sort_m=np.zeros([1,nRows])
        sort_m=sort_network_m[i,:]
        idx_m=np.zeros([1,nRows])
        idx_m=idx_network_m[i,:]
#            -np.sort(-knn_network_m, axis=0)
#        idx_m=np.argsort(-knn_network_m, axis=0)
        for k in range(0,K):
            sum_m=np.sum(sort_m[k])

        for j in range(0, K):
            if j==0:
                w[0,0]=sort_m[0]
            else:
                w[0,j]=r**(j-1)*sort_m[j]

            Y_m[:,i]=Y_m[:,i]+w[0,j]*known_interaction[:,idx_m[j]]

        if (sum_m!=0):
            Y_m[:,i]=Y_m[:,i]/sum_m
        else:
            Y_m[:,i]=Y_m[:,i]/k

    knn_network_d,sort_network_d, idx_network_d = KNN(diseaseSM, K)  # for diseases
    for i in range(0, nRows):
        w2 = np.zeros([1, K])
        sort_d = sort_network_d[i,:]
        idx_d = idx_network_d[i,:]
        for k in range(0, K):
            sum_d = np.sum(sort_d[ k])

        for j1 in range(0, K):
            if j1==0:
                w[0,0]=sort_d[j1]
            else:
                w[0, j1] = r ** (j1 - 1) * sort_d[ j1]

            Y_d[i,:] = Y_d[i,:] + w[0, j1] * known_interaction[idx_d[ j],:]
        if (sum_d!=0):
            Y_d[i,:] = Y_d[i,:] / sum_d
        else:
            Y_d[i, :]=Y_d[i,:] / k

    a1=1
    a2=1
    Y_md=(Y_m*a1+ Y_d*a2)/(a1+a2)
    MD_mat_new=np.zeros([nRows, nCols])
    for i in range(0, nRows):
        for j in range (0, nCols):
            MD_mat_new[i,j]=max(known_interaction[i,j], Y_md[i,j])
            if (MD_mat_new[i,j]>1):
                MD_mat_new[i, j]=1
    #
    # for i in range(0, nRows):
    #     for j in range (0, nCols):
    #         MD_mat_new[i,j]=max(known_interaction[i,j], Y_md[i,j])
    return MD_mat_new

A=np.array(pd.read_csv("CDataset_DiDrA.csv", header=None))
print(A.shape)

known_interaction=A
diseaseSimilarity=np.array(pd.read_csv("CDataset_DiseaseSim.csv", header=None))
DrugSimilarity=np.array(pd.read_csv("CDataset_DrugSim.csv", header=None))

A87=CalWeightedKNN(known_interaction,diseaseSimilarity,DrugSimilarity,7,0.7)
print("Kich thuoc ma tran A1 la", A87.shape)
count87=0
for i in range(0,A87.shape[0]):
    for j in range(0, A87.shape[1]):
        if (A87[i,j]>0.2):
            A87[i,j]=1
            count87=count87+1
        else:
            A87[i,j]=0

print('So quan he Drug-Disease moi =', count87)
A87=A87
print('Kich thuoc ma tran moi', A87.shape)
np.savetxt('UpdateDrugDiseaseAssociationMatrix7r07.csv', A87.astype(float), delimiter=",")
