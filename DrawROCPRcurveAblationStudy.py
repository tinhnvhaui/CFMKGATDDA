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
y_real = np.array(pd.read_csv("New_lncRNADiseasematrix_General_30032024.csv", header=None))
y_real2 = np.array(pd.read_csv("05-lncRNA-disease.csv", header=None))
y_proba1 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold6_10052024.csv", header=None))
y_proba2 = np.array(pd.read_csv("PredictedResult_General_withoutIntegratedSM_30032024.csv", header=None))
y_proba3 = np.array(pd.read_csv("PredictedResult_lncRNADisease_matrix_withoutIntegrated_withoutCF.csv", header=None))
#y_proba4 = np.array(pd.read_csv("PredictedResult_lncRNADisease_matrix_withIntegrated_withoutCF.csv", header=None))

AUC = []
ACC = []
RECALL = []
PREECISION = []
AUPR = []
F1 = []

FPR = []
TPR = []
THR1 = []

THR2 = []
PRE = []
REC = []
auc = roc_auc_score(y_real, y_proba1)
fpr, tpr, thresholds1 = metrics.roc_curve(y_real.ravel(), y_proba1.ravel(),
                                              pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR.append(fpr)
TPR.append(tpr)
THR1.append(thresholds1)
AUC.append(auc)
precision, recall, thresholds2 = precision_recall_curve(y_real.ravel(), y_proba1.ravel())
aupr = metrics.auc(recall, precision)  #
AUPR.append(aupr)
THR2.append(thresholds2)
PRE.append(precision)
REC.append(recall)
# ROC curve
plt.figure()
tprs=[]
mean_fpr=np.linspace(0,1,1000)
for i in range(len(FPR)):
    tprs.append(np.interp(mean_fpr,FPR[i],TPR[i]))
    tprs[-1][0]=0.0
    auc = metrics.auc(FPR[i], TPR[i])
    print('auc=',auc)
#     plt.xlim(0, 1)  #
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR[i], TPR[i],lw=2,label='CFGANLDA ROC (Auc=%0.4f)' % auc)
    plt.legend()  #

#ROC fold 2
AUC2 = []
ACC2 = []
RECALL2 = []
PREECISION2 = []
AUPR2 = []
F12 = []

FPR2 = []
TPR2 = []
THR12 = []

THR22 = []
PRE2 = []
REC2 = []
auc2 = roc_auc_score(y_real, y_proba2)
fpr2, tpr2, thresholds12 = metrics.roc_curve(y_real.ravel(), y_proba2.ravel(),
                                              pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR2.append(fpr2)
TPR2.append(tpr2)
THR12.append(thresholds12)
AUC2.append(auc2)
precision2, recall2, thresholds22 = precision_recall_curve(y_real.ravel(), y_proba2.ravel())
aupr2 = metrics.auc(recall2, precision2)  #
AUPR2.append(aupr2)
THR22.append(thresholds22)
PRE2.append(precision2)
REC2.append(recall2)
# ROC curve
#plt.figure()
tprs2=[]
mean_fpr2=np.linspace(0,1,1000)
for i in range(len(FPR2)):
    tprs2.append(np.interp(mean_fpr2,FPR2[i],TPR2[i]))
    tprs2[-1][0]=0.0
    auc2 = metrics.auc(FPR2[i], TPR2[i])
    print('auc=',auc2)
#     plt.xlim(0, 1)  #
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR2[i], TPR2[i],lw=0.7,label='No-integrated SM, with CF case-ROC (Auc=%0.4f)' % auc2)
    plt.legend()  #
 # ROC fold 3

AUC3 = []
ACC3 = []
RECALL3 = []
PREECISION3 = []
AUPR3 = []
F13 = []

FPR3 = []
TPR3 = []
THR13 = []

THR23 = []
PRE3 = []
REC3 = []
auc3 = roc_auc_score(y_real2, y_proba3)
fpr3, tpr3, thresholds13 = metrics.roc_curve(y_real2.ravel(), y_proba3.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR3.append(fpr3)
TPR3.append(tpr3)
THR13.append(thresholds13)
AUC3.append(auc3)
precision3, recall3, thresholds23 = precision_recall_curve(y_real.ravel(), y_proba3.ravel())
aupr3 = metrics.auc(recall3, precision3)  #
AUPR3.append(aupr3)
THR23.append(thresholds23)
PRE3.append(precision3)
REC3.append(recall3)
# ROC curve
#plt.figure()
tprs3 = []
mean_fpr3 = np.linspace(0, 1, 1000)
for i in range(len(FPR3)):
    tprs3.append(np.interp(mean_fpr3, FPR3[i], TPR3[i]))
    tprs3[-1][0] = 0.0
    auc3 = metrics.auc(FPR3[i], TPR3[i])
    print('auc=', auc3)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR3[i], TPR3[i], lw=0.7, label='No-integrated SM and no CF case-ROC (Auc=%0.4f)' % auc3)
    plt.legend()  #
# # ROC fold 4
# AUC4 = []
# ACC4 = []
# RECALL4 = []
# PREECISION4 = []
# AUPR4 = []
# F14 = []
#
# FPR4 = []
# TPR4 = []
# THR14 = []
#
# THR24 = []
# PRE4 = []
# REC4 = []
# auc4 = roc_auc_score(y_real2, y_proba4)
# fpr4, tpr4, thresholds14 = metrics.roc_curve(y_real.ravel(), y_proba4.ravel(),
#                                              pos_label=1)  # The actual value indicated as 1 is a positive sample.
# FPR4.append(fpr4)
# TPR4.append(tpr4)
# THR14.append(thresholds14)
# AUC4.append(auc4)
# precision4, recall4, thresholds24 = precision_recall_curve(y_real.ravel(), y_proba4.ravel())
# aupr4 = metrics.auc(recall4, precision4)  #
# AUPR4.append(aupr4)
# THR24.append(thresholds24)
# PRE4.append(precision4)
# REC4.append(recall4)
# # ROC curve
# #plt.figure()
# tprs4 = []
# mean_fpr4 = np.linspace(0, 1, 1000)
# for i in range(len(FPR4)):
#     tprs4.append(np.interp(mean_fpr4, FPR4[i], TPR4[i]))
#     tprs4[-1][0] = 0.0
#     auc4 = metrics.auc(FPR4[i], TPR4[i])
#     print('auc=', auc4)
#     #     plt.xlim(0, 1)  #
#     #     plt.ylim(0, 1)  #
#     plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#     plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#     plt.plot(FPR4[i], TPR4[i], lw=0.7, label='Integrated similarity and no CF case-ROC (Auc=%0.4f)' % auc4)
#     plt.legend()  #

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#
plt.title("Receiver Operating Characteristic Curve")
plt.legend(loc="lower right")
#plt.legend(bbox_to_anchor = (1.05, 0), loc=3, borderaxespad = 0)#
# plt.legend()

plt.show()
plt.figure()
for i in range(len(REC)):
    aupr = metrics.auc(REC[i], PRE[i])
    print('AUPR=',aupr)
#     print(auc)
#     plt.xlim(0, 1)  #
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC[i], PRE[i],lw=2,label='CFGANLDA PR curve (Aupr=%0.4f)' % aupr)
    plt.legend()  #

# Precision-recall curve fold 2
for i in range(len(REC2)):
    aupr2 = metrics.auc(REC2[i], PRE2[i])
    print('AUPR=',aupr2)
#     print(auc)
#     plt.xlim(0, 1)  #
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC2[i], PRE2[i],lw=0.7,label='No-integrated SM and with CF-PR curve (Aupr=%0.4f)' % aupr2)
    plt.legend()  #

# Precision-recall curve fold 3
for i in range(len(REC3)):
    aupr3 = metrics.auc(REC3[i], PRE3[i])
    print('AUPR=',aupr3)
#     print(auc)
#     plt.xlim(0, 1)  #
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC3[i], PRE3[i],lw=0.7,label='No-integrated SM and no CF-PR curve (Aupr=%0.4f)' % aupr3)
    plt.legend()  #

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="center")
plt.show()