import csv
import numpy as np
import pandas as pd
import scipy.odr as sci
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
import math
from sklearn import preprocessing as pp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

#print(left, right)
matrix11=np.array(pd.read_csv("5fold/fold01/UpdatedAdjacencyMatrix_K05r07_5fold_fold01_30032021.csv", header=None))
matrix12=np.array(pd.read_csv("5fold/fold01/Adjacency_5fold_fold01_30032021.csv", header=None))
matrix13=np.array(pd.read_csv("5fold/fold01/Adjacency_5fold_fold01_30032021.csv", header=None))
matrix14=np.array(pd.read_csv("5fold/fold01/UpdatedAdjacencyMatrix_K05r07_5fold_fold01_30032021.csv", header=None))
matrix15=np.array(pd.read_csv("miRNA_diseaseassociationmatrix.csv", header=None))
print(' kich thuoc ma tran la 1', matrix11.shape)
A11=matrix11
for i in range(0,A11.shape[0]):
   for j in range(0, A11.shape[1]):
       if (A11[i,j]>0):
           A11[i,j]=1
       else:
           A11[i, j]=0

A12=matrix12
for i in range(0,A12.shape[0]):
   for j in range(0, A12.shape[1]):
       if (A12[i,j]>0):
           A12[i,j]=1
       else:
           A12[i, j]=0

A13=matrix13
for i in range(0,A13.shape[0]):
   for j in range(0, A13.shape[1]):
       if (A13[i,j]>0):
           A13[i,j]=1
       else:
           A13[i, j]=0

A14=matrix14
for i in range(0,A14.shape[0]):
   for j in range(0, A14.shape[1]):
       if (A14[i,j]>0):
           A14[i,j]=1
       else:
           A14[i, j]=0

A15=matrix15
for i in range(0,A15.shape[0]):
   for j in range(0, A15.shape[1]):
       if (A15[i,j]>0):
           A15[i,j]=1
       else:
           A15[i, j]=0
Y_true=A11
Y_true1=A12
Y_true2=A13
Y_true3=A14
Y_true4=A15
# A12=matrix12
# Y_true12=A12

# new method
Y_pred11=np.array(pd.read_csv("5fold/fold01/FinalPredictionScore_30032021_5fold_fold01_1.csv", header=None))
Y_pred12=np.array(pd.read_csv("5fold/fold01/FinalPredictionScore_30032021_5fold_fold01_2.csv", header=None))
# Y_pred=(Y_pred11+Y_pred12)/2
Y_pred=Y_pred11
# No WKNKN-noIN
print(' Ma tran ban dau la', Y_true)
Y_pred111=np.array(pd.read_csv("5fold/fold01/noWKNKN_noIn/FinalPredictionScore_31032021_5fold_fold01_2_withoutWKNKN.csv", header=None))
Y_pred112=np.array(pd.read_csv("5fold/fold01/noWKNKN_noIn/FinalPredictionScore_31032021_5fold_fold01_1_withoutWKNKN.csv", header=None))
#Y_pred2=np.array(pd.read_csv("general/FinalPredictionScore_1general_17032021_2.csv", header=None))
#Y_pred=Y_pred2
#Y_pred1=(Y_pred111+Y_pred112)/2
Y_pred1=Y_pred111
# No WKNKN-withIN
Y_pred121=np.array(pd.read_csv("5fold/fold01/noW_withIn/FinalPredictionScore_31032021_5fold_fold01_noWKNKN_withIn_1.csv", header=None))
Y_pred122=np.array(pd.read_csv("5fold/fold01/noW_withIn/FinalPredictionScore_31032021_5fold_fold01_noWKNKN_withIn_2.csv", header=None))
#Y_pred2=np.array(pd.read_csv("general/FinalPredictionScore_1general_17032021_2.csv", header=None))
#Y_pred=Y_pred2
Y_pred2=Y_pred122

#
# with WKNKN-noIN
Y_pred131=np.array(pd.read_csv("5fold/fold01/withW_noIn/FinalPredictionScore_31032021_5fold_fold01_withWKNKN_noIn.csv", header=None))
Y_pred132=np.array(pd.read_csv("5fold/fold01/withW_noIn/FinalPredictionScore_31032021_5fold_fold01_2_withWKNKN_noIn.csv", header=None))
#Y_pred2=np.array(pd.read_csv("general/FinalPredictionScore_1general_17032021_2.csv", header=None))
#Y_pred=Y_pred2
Y_pred3=Y_pred131
#Y_pred3=(Y_pred131+Y_pred132)/2
# PMFMDA
# Y_pred141=np.array(pd.read_csv("5fold/fold04/FinalPredictionScore_30032021_5fold_fold04_1.csv", header=None))
# Y_pred142=np.array(pd.read_csv("5fold/fold04/FinalPredictionScore_30032021_5fold_fold04_2.csv", header=None))
# #Y_pred2=np.array(pd.read_csv("general/FinalPredictionScore_1general_17032021_2.csv", header=None))
#Y_pred=Y_pred2
Y_pred4=np.array(pd.read_csv("general/PMFMDA/final_score.csv",header=None))


#Y_pred2=np.array(pd.read_csv("general/FinalPredictionScore_1general_17032021_2.csv", header=None))
#Y_pred=Y_pred2
#Y_pred4=(Y_pred151+Y_pred152)/2
# Tinh AUC fold1
print('Ma tran du doan la', Y_pred)
print('Min max scaling')
data_mms = Y_pred
mms = pp.MinMaxScaler()
data_mms = mms.fit_transform(Y_pred)

# Tinh AUC fold2
print('Ma tran du doan la', Y_pred1)
print('Min max scaling')
data_mms1 = Y_pred1
mms = pp.MinMaxScaler()
data_mms1 = mms.fit_transform(Y_pred1)

# Tinh AUC fold3
print('Ma tran du doan la', Y_pred1)
print('Min max scaling')
data_mms2 = Y_pred2
mms = pp.MinMaxScaler()
data_mms2 = mms.fit_transform(Y_pred2)

# Tinh AUC fold4
print('Ma tran du doan la', Y_pred3)
print('Min max scaling')
data_mms3 = Y_pred3
mms = pp.MinMaxScaler()
data_mms3 = mms.fit_transform(Y_pred3)

# Tinh AUC fold5
print('Ma tran du doan la', Y_pred1)
print('Min max scaling')
data_mms4 = Y_pred4
# mms = pp.MinMaxScaler()
# data_mms4 = mms.fit_transform(Y_pred4)

# # Luu file ket qua cuoi cung, lam co so check case studies
# np.savetxt("general/FinalPredictionScore_general_final_forcheckcasestudies_3003.txt", data_mms.astype(float))
# print('Kich thuoc ma tran du doan sau chuan hoa la:', data_mms.shape)
# print('Ma tran du doan sau chuan hoa la la', data_mms)

# Ve do thi cho phuong phap moi
n_classes=Y_pred.shape[1]
fpr = dict()
pr = dict()
tpr = dict()
#fnr=dict()
#tnr=dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i],_= roc_curve(Y_true[:, i], data_mms[:, i],pos_label=None)
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["macro"], tpr["macro"],_= roc_curve(Y_true.ravel(), data_mms.ravel(), pos_label=None)
roc_auc_final = roc_auc_score(Y_true.ravel(),data_mms.ravel())

# tinh precision, recall cho phuong phap moi
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i],_ = precision_recall_curve(Y_true[:, i],data_mms[:, i])
    average_precision[i] = average_precision_score(Y_true[:, i],data_mms[:, i])

precision["macro"], recall["macro"], _ = precision_recall_curve(Y_true.ravel(), data_mms.ravel())
average_precision["macro"] = average_precision_score(Y_true.ravel(), data_mms.ravel())
# print('Precision-Recall Curve over all classes: {0:0.6f}'
#       .format(average_precision["macro"]))
#
# print('roc_auc=', roc_auc_final)
# lw = 2
# plt.plot(fpr["macro"], tpr["macro"], color='darkorange',
#          lw=lw, label='RWRIMSMDA (AUC = %0.6f)' % roc_auc_final)


# No WKNKN_no Integrate
n_classes_1=Y_pred1.shape[1]
fpr_1 = dict()
pr_1 = dict()
tpr_1 = dict()
#fnr=dict()
#tnr=dict()
roc_auc_1 = dict()
for i in range(n_classes_1):
    fpr_1[i], tpr_1[i],_= roc_curve(Y_true1[:, i], data_mms1[:, i],pos_label=None)
    roc_auc_1[i] = auc(fpr_1[i], tpr_1[i])

fpr_1["macro"], tpr_1["macro"],_= roc_curve(Y_true1.ravel(), data_mms1.ravel(), pos_label=None)
roc_auc_final_1 = roc_auc_score(Y_true1.ravel(),data_mms1.ravel())

# tinh precision, recall cho truong hop no WKNKN no Integrate
precision_1 = dict()
recall_1 = dict()
average_precision_1 = dict()
for i in range(n_classes_1):
    precision_1[i], recall_1[i],_ = precision_recall_curve(Y_true1[:, i],data_mms1[:, i])
    average_precision_1[i] = average_precision_score(Y_true1[:, i],data_mms1[:, i])

precision_1["macro"], recall_1["macro"], _ = precision_recall_curve(Y_true1.ravel(),data_mms1.ravel())
average_precision_1["macro"] = average_precision_score(Y_true1.ravel(),data_mms1.ravel())

# No WKNKN_with Integrate
n_classes_2=Y_pred2.shape[1]
fpr_2 = dict()
pr_2 = dict()
tpr_2 = dict()
#fnr=dict()
#tnr=dict()
roc_auc_2 = dict()
for i in range(n_classes_2):
    fpr_2[i], tpr_2[i],_= roc_curve(Y_true2[:, i], data_mms2[:, i],pos_label=None)
    roc_auc_2[i] = auc(fpr_2[i], tpr_2[i])

fpr_2["macro"], tpr_2["macro"],_= roc_curve(Y_true2.ravel(), data_mms2.ravel(), pos_label=None)
roc_auc_final_2 = roc_auc_score(Y_true2.ravel(),data_mms2.ravel())

# tinh precision, recall cho truong hop No WKNKN_with Integrate
precision_2 = dict()
recall_2 = dict()
average_precision_2 = dict()
for i in range(n_classes_2):
    precision_2[i], recall_2[i],_ = precision_recall_curve(Y_true2[:, i],data_mms2[:, i])
    average_precision_2[i] = average_precision_score(Y_true2[:, i],data_mms2[:, i])

precision_2["macro"], recall_2["macro"], _ = precision_recall_curve(Y_true2.ravel(),data_mms2.ravel())
average_precision_2["macro"] = average_precision_score(Y_true2.ravel(),data_mms2.ravel())

# with WKNKN_ No Integrate
n_classes_3=Y_pred3.shape[1]
fpr_3 = dict()
pr_3 = dict()
tpr_3 = dict()
#fnr=dict()
#tnr=dict()
roc_auc_3 = dict()
for i in range(n_classes_3):
    fpr_3[i], tpr_3[i],_= roc_curve(Y_true3[:, i], data_mms3[:, i],pos_label=None)
    roc_auc_3[i] = auc(fpr_3[i], tpr_3[i])

fpr_3["macro"], tpr_3["macro"],_= roc_curve(Y_true3.ravel(), data_mms3.ravel(), pos_label=None)
roc_auc_final_3 = roc_auc_score(Y_true3.ravel(),data_mms3.ravel())

# tinh precision, recall cho phuong phap moi
precision_3 = dict()
recall_3 = dict()
average_precision_3 = dict()
for i in range(n_classes_3):
    precision_3[i], recall_3[i],_ = precision_recall_curve(Y_true3[:, i],data_mms3[:, i])
    average_precision_3[i] = average_precision_score(Y_true3[:, i],data_mms3[:, i])

precision_3["macro"], recall_3["macro"], _ = precision_recall_curve(Y_true3.ravel(), data_mms3.ravel())
average_precision_3["macro"] = average_precision_score(Y_true3.ravel(), data_mms3.ravel())

# PMFMDA
n_classes_4=Y_pred4.shape[1]
fpr_4 = dict()
pr_4 = dict()
tpr_4 = dict()
#fnr=dict()
#tnr=dict()
roc_auc_4 = dict()
for i in range(n_classes_4):
    fpr_4[i], tpr_4[i],_= roc_curve(Y_true4[:, i], data_mms4[:, i],pos_label=None)
    roc_auc_4[i] = auc(fpr_4[i], tpr_4[i])

fpr_4["macro"], tpr_4["macro"],_= roc_curve(Y_true4.ravel(), data_mms4.ravel(), pos_label=None)
roc_auc_final_4 = roc_auc_score(Y_true4.ravel(),data_mms4.ravel())

# tinh precision, recall cho phuong phap moi
precision_4 = dict()
recall_4 = dict()
average_precision_4 = dict()
for i in range(n_classes_4):
    precision_4[i], recall_4[i],_ = precision_recall_curve(Y_true4[:, i],data_mms4[:, i])
    average_precision_4[i] = average_precision_score(Y_true4[:, i],data_mms4[:, i])

precision_4["macro"], recall_4["macro"], _ = precision_recall_curve(Y_true4.ravel(), data_mms4.ravel())
average_precision_4["macro"] = average_precision_score(Y_true4.ravel(), data_mms4.ravel())

# ve do thi Phuong phap moi
lw = 1
plt.plot(fpr["macro"], tpr["macro"], color='darkorange',
         lw=lw, label='RWRMMDA (AUC = %0.6f)' % roc_auc_final)

# ve do thi AUC noWKNKN-no integrate
plt.plot(fpr_1["macro"], tpr_1["macro"], color='red',
         lw=lw, label='NTSHMDA (AUC = %0.6f)' % roc_auc_final_1)

# # ve do thi AUC noWKNKN-with integrate
# plt.plot(fpr_2["macro"], tpr_2["macro"], color='blue',
#          lw=lw, label='RWRMMDA in case of using Integrated similarities only (AUC = %0.6f)' % roc_auc_final_2)
#
# # ve do thi AUC with WKNKN-with integrate
# plt.plot(fpr_3["macro"], tpr_3["macro"], color='green',
#          lw=lw, label='RWRMMDA in case of using WKNKN only (AUC = %0.6f)' % roc_auc_final_3)

#ve do thi AUC PMFMDA
plt.plot(fpr_4["macro"], tpr_4["macro"], color='black',
         lw=lw, label='PMFMDA (AUC = %0.6f)' % roc_auc_final_4)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves and AUC values in comparison with other related methods ')
#plt.title('ROC curves and AUC values in different cases of RWRMMDAs ')
plt.legend(loc="lower right")
plt.show()


plt.step(recall['macro'], precision['macro'], where='post', label='RWRMMDA: AUPR={0:0.6f}'.format(average_precision["macro"]))
plt.step(recall_1['macro'], precision_1['macro'], where='post', label='NTSHMDA: AUPR={0:0.6f}'.format(average_precision_1["macro"]))
#plt.step(recall_2['macro'], precision_2['macro'], where='post', label='RWRMMDA in case of using Integrated similarities only: AUPR={0:0.6f}'.format(average_precision_2["macro"]))
#plt.step(recall_3['macro'], precision_3['macro'], where='post', label='RWRMMDA in case of using WKNKN only: AUPR={0:0.6f}'.format(average_precision_3["macro"]))
plt.step(recall_4['macro'], precision_4['macro'], where='post', label='PMFMDA: AUPR={0:0.6f}'.format(average_precision_4["macro"]))
plt.title('Precision recall curves and AUPR values in comparison with other related methods')
#plt.title('Precision recall curves and AUPR values in different cases of RWRMMDAs')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.title('Precision recall curve for ITPGLDA')
plt.legend(loc="lower right")
plt.show()
#plt.savefig('AUPR_dif_cases.png')
