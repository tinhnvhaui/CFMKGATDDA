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
y_real = np.array(pd.read_csv("New_lncRNADiseasematrix_General_30032024.csv", header=None))
y_proba1 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold1_10052024.csv", header=None))
y_proba2 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold2_10052024.csv", header=None))
y_proba3 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold3_10052024.csv", header=None))
y_proba4 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold4_10052024.csv", header=None))
y_proba5 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold5_10052024.csv", header=None))
y_proba6 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold6_10052024.csv", header=None))
y_proba7 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold7_10052024.csv", header=None))
y_proba8 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold6_11042024.csv", header=None))
y_proba9 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold9_10052024.csv", header=None))
y_proba10 = np.array(pd.read_csv("NewPredictedResult_matrix_10fold_fold10_10052024.csv", header=None))

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
    plt.plot(FPR[i], TPR[i],lw=0.7,label='ROC fold 1 (Auc=%0.4f)' % auc)
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
    plt.plot(FPR2[i], TPR2[i],lw=0.7,label='ROC fold 2 (Auc=%0.4f)' % auc2)
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
auc3 = roc_auc_score(y_real, y_proba3)
fpr3, tpr3, thresholds13 = metrics.roc_curve(y_real.ravel(), y_proba3.ravel(),
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
    plt.plot(FPR3[i], TPR3[i], lw=0.7, label='ROC fold 3 (Auc=%0.4f)' % auc3)
    plt.legend()  #
# ROC fold 4
AUC4 = []
ACC4 = []
RECALL4 = []
PREECISION4 = []
AUPR4 = []
F14 = []

FPR4 = []
TPR4 = []
THR14 = []

THR24 = []
PRE4 = []
REC4 = []
auc4 = roc_auc_score(y_real, y_proba4)
fpr4, tpr4, thresholds14 = metrics.roc_curve(y_real.ravel(), y_proba4.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR4.append(fpr4)
TPR4.append(tpr4)
THR14.append(thresholds14)
AUC4.append(auc4)
precision4, recall4, thresholds24 = precision_recall_curve(y_real.ravel(), y_proba4.ravel())
aupr4 = metrics.auc(recall4, precision4)  #
AUPR4.append(aupr4)
THR24.append(thresholds24)
PRE4.append(precision4)
REC4.append(recall4)
# ROC curve
#plt.figure()
tprs4 = []
mean_fpr4 = np.linspace(0, 1, 1000)
for i in range(len(FPR4)):
    tprs4.append(np.interp(mean_fpr4, FPR4[i], TPR4[i]))
    tprs4[-1][0] = 0.0
    auc4 = metrics.auc(FPR4[i], TPR4[i])
    print('auc=', auc4)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR4[i], TPR4[i], lw=0.7, label='ROC fold 4 (Auc=%0.4f)' % auc4)
    plt.legend()  #

# ROC fold 5
AUC5 = []
ACC5 = []
RECALL5 = []
PREECISION5 = []
AUPR5 = []
F15 = []

FPR5 = []
TPR5 = []
THR15 = []

THR25 = []
PRE5 = []
REC5 = []
auc5 = roc_auc_score(y_real, y_proba5)
fpr5, tpr5, thresholds15 = metrics.roc_curve(y_real.ravel(), y_proba5.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR5.append(fpr5)
TPR5.append(tpr5)
THR15.append(thresholds15)
AUC5.append(auc5)
precision5, recall5, thresholds25 = precision_recall_curve(y_real.ravel(), y_proba5.ravel())
aupr5 = metrics.auc(recall5, precision5)  #
AUPR5.append(aupr5)
THR25.append(thresholds25)
PRE5.append(precision5)
REC5.append(recall5)
# ROC curve
#plt.figure()
tprs5 = []
mean_fpr5 = np.linspace(0, 1, 1000)
for i in range(len(FPR5)):
    tprs5.append(np.interp(mean_fpr5, FPR5[i], TPR5[i]))
    tprs5[-1][0] = 0.0
    auc5 = metrics.auc(FPR5[i], TPR5[i])
    print('auc=', auc5)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR5[i], TPR5[i], lw=0.7, label='ROC fold 5 (Auc=%0.4f)' % auc5)
    plt.legend()  #

# ROC fold 6
AUC6 = []
ACC6 = []
RECALL6 = []
PREECISION6 = []
AUPR6 = []
F16 = []

FPR6 = []
TPR6 = []
THR16 = []

THR26 = []
PRE6 = []
REC6 = []
auc6 = roc_auc_score(y_real, y_proba6)
fpr6, tpr6, thresholds16 = metrics.roc_curve(y_real.ravel(), y_proba6.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR6.append(fpr6)
TPR6.append(tpr6)
THR16.append(thresholds16)
AUC6.append(auc6)
precision6, recall6, thresholds26 = precision_recall_curve(y_real.ravel(), y_proba6.ravel())
aupr6 = metrics.auc(recall6, precision6)  #
AUPR6.append(aupr6)
THR26.append(thresholds26)
PRE6.append(precision6)
REC6.append(recall6)
# ROC curve
#plt.figure()
tprs6 = []
mean_fpr6 = np.linspace(0, 1, 1000)
for i in range(len(FPR6)):
    tprs6.append(np.interp(mean_fpr6, FPR6[i], TPR6[i]))
    tprs6[-1][0] = 0.0
    auc6 = metrics.auc(FPR6[i], TPR6[i])
    print('auc=', auc6)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR6[i], TPR6[i], lw=0.7, label='ROC fold 6 (Auc=%0.4f)' % auc6)
    plt.legend()  #

# ROC fold 7
AUC7 = []
ACC7 = []
RECALL7 = []
PREECISION7 = []
AUPR7 = []
F17 = []

FPR7 = []
TPR7 = []
THR17 = []

THR27 = []
PRE7 = []
REC7 = []
auc7 = roc_auc_score(y_real, y_proba7)
fpr7, tpr7, thresholds17 = metrics.roc_curve(y_real.ravel(), y_proba7.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR7.append(fpr7)
TPR7.append(tpr7)
THR17.append(thresholds17)
AUC7.append(auc7)
precision7, recall7, thresholds27 = precision_recall_curve(y_real.ravel(), y_proba7.ravel())
aupr7 = metrics.auc(recall7, precision7)  #
AUPR7.append(aupr7)
THR27.append(thresholds27)
PRE7.append(precision7)
REC7.append(recall7)
# ROC curve
#plt.figure()
tprs7 = []
mean_fpr7 = np.linspace(0, 1, 1000)
for i in range(len(FPR7)):
    tprs7.append(np.interp(mean_fpr7, FPR7[i], TPR7[i]))
    tprs7[-1][0] = 0.0
    auc7 = metrics.auc(FPR7[i], TPR7[i])
    print('auc=', auc7)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR7[i], TPR7[i], lw=0.7, label='ROC fold 7 (Auc=%0.4f)' % auc7)
    plt.legend()  #

# ROC fold 8
AUC8 = []
ACC8 = []
RECALL8 = []
PREECISION8 = []
AUPR8 = []
F18 = []

FPR8 = []
TPR8 = []
THR18 = []

THR28 = []
PRE8 = []
REC8 = []
auc8 = roc_auc_score(y_real, y_proba8)
fpr8, tpr8, thresholds18 = metrics.roc_curve(y_real.ravel(), y_proba8.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR8.append(fpr8)
TPR8.append(tpr8)
THR18.append(thresholds18)
AUC8.append(auc8)
precision8, recall8, thresholds28 = precision_recall_curve(y_real.ravel(), y_proba8.ravel())
aupr8 = metrics.auc(recall8, precision8)  #
AUPR8.append(aupr8)
THR28.append(thresholds28)
PRE8.append(precision8)
REC8.append(recall8)
# ROC curve
#plt.figure()
tprs8 = []
mean_fpr8 = np.linspace(0, 1, 1000)
for i in range(len(FPR8)):
    tprs8.append(np.interp(mean_fpr8, FPR8[i], TPR8[i]))
    tprs8[-1][0] = 0.0
    auc8 = metrics.auc(FPR8[i], TPR8[i])
    print('auc=', auc8)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR8[i], TPR8[i], lw=0.7, label='ROC fold 8 (Auc=%0.4f)' % auc8)
    plt.legend()  #


# ROC fold 9
AUC9 = []
ACC9 = []
RECALL9 = []
PREECISION9 = []
AUPR9 = []
F19 = []

FPR9 = []
TPR9 = []
THR19 = []

THR29 = []
PRE9 = []
REC9 = []
auc9 = roc_auc_score(y_real, y_proba9)
fpr9, tpr9, thresholds19 = metrics.roc_curve(y_real.ravel(), y_proba9.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR9.append(fpr9)
TPR9.append(tpr9)
THR19.append(thresholds19)
AUC9.append(auc9)
precision9, recall9, thresholds29 = precision_recall_curve(y_real.ravel(), y_proba9.ravel())
aupr9 = metrics.auc(recall9, precision9)  #
AUPR9.append(aupr9)
THR29.append(thresholds29)
PRE9.append(precision9)
REC9.append(recall9)
# ROC curve
#plt.figure()
tprs9 = []
mean_fpr9 = np.linspace(0, 1, 1000)
for i in range(len(FPR9)):
    tprs9.append(np.interp(mean_fpr9, FPR9[i], TPR9[i]))
    tprs9[-1][0] = 0.0
    auc9 = metrics.auc(FPR9[i], TPR9[i])
    print('auc=', auc9)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR9[i], TPR9[i], lw=0.7, label='ROC fold 9 (Auc=%0.4f)' % auc9)
    plt.legend()  #

# ROC fold 10
AUC10 = []
ACC10 = []
RECALL10 = []
PREECISION10 = []
AUPR10 = []
F110 = []

FPR10 = []
TPR10 = []
THR110 = []

THR210 = []
PRE10 = []
REC10 = []
auc10 = roc_auc_score(y_real, y_proba10)
fpr10, tpr10, thresholds110 = metrics.roc_curve(y_real.ravel(), y_proba10.ravel(),
                                             pos_label=1)  # The actual value indicated as 1 is a positive sample.
FPR10.append(fpr10)
TPR10.append(tpr10)
THR110.append(thresholds110)
AUC10.append(auc10)
precision10, recall10, thresholds210 = precision_recall_curve(y_real.ravel(), y_proba10.ravel())
aupr10 = metrics.auc(recall10, precision10)  #
AUPR10.append(aupr10)
THR210.append(thresholds210)
PRE10.append(precision10)
REC10.append(recall10)
# ROC curve
#plt.figure()
tprs10 = []
mean_fpr10 = np.linspace(0, 1, 1000)
for i in range(len(FPR10)):
    tprs10.append(np.interp(mean_fpr10, FPR10[i], TPR10[i]))
    tprs10[-1][0] = 0.0
    auc10 = metrics.auc(FPR10[i], TPR10[i])
    print('auc=', auc10)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR10[i], TPR10[i], lw=0.7, label='ROC fold 10 (Auc=%0.4f)' % auc10)
    plt.legend()  #
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
    plt.plot(REC[i], PRE[i],lw=0.7,label='Precision-Recall fold 1(Aupr=%0.4f)' % aupr)
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
    plt.plot(REC2[i], PRE2[i],lw=0.7,label='Precision-Recall fold 2 (Aupr=%0.4f)' % aupr2)
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
    plt.plot(REC3[i], PRE3[i],lw=0.7,label='Precision-Recall fold 3 (Aupr=%0.4f)' % aupr3)
    plt.legend()  #
# Precision-recall curve fold 4
for i in range(len(REC4)):
    aupr4 = metrics.auc(REC4[i], PRE4[i])
    print('AUPR=', aupr4)
    #     print(auc)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC4[i], PRE4[i], lw=0.7, label='Precision-Recall fold 4 (Aupr=%0.4f)' % aupr4)
    plt.legend()  #

# Precision-recall curve fold 5
for i in range(len(REC5)):
    aupr5 = metrics.auc(REC5[i], PRE5[i])
    print('AUPR=', aupr5)
    #     print(auc)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC5[i], PRE5[i], lw=0.7, label='Precision-Recall fold 5 (Aupr=%0.4f)' % aupr5)
    plt.legend()  #
# Precision-recall curve fold 6
for i in range(len(REC6)):
    aupr6 = metrics.auc(REC6[i], PRE6[i])
    print('AUPR=', aupr6)
    #     print(auc)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC6[i], PRE6[i], lw=0.7, label='Precision-Recall fold 6 (Aupr=%0.4f)' % aupr6)
    plt.legend()  #
# Precision-recall curve fold 7
for i in range(len(REC7)):
    aupr7 = metrics.auc(REC7[i], PRE7[i])
    print('AUPR=', aupr7)
    #     print(auc)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC7[i], PRE7[i], lw=0.7, label='Precision-Recall fold 7 (Aupr=%0.4f)' % aupr7)
    plt.legend()  #
# Precision-recall curve fold 8
# Precision-recall curve fold 8
for i in range(len(REC8)):
    aupr8 = metrics.auc(REC8[i], PRE8[i])
    print('AUPR=', aupr8)
    #     print(auc)
    #     plt.xlim(0, 1)  #
    #     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC8[i], PRE8[i], lw=0.7, label='Precision-Recall fold 8 (Aupr=%0.4f)' % aupr8)
    plt.legend()  #
# Precision-recall curve fold 9
for i in range(len(REC9)):
    aupr9 = metrics.auc(REC9[i], PRE9[i])
    print('AUPR=', aupr9)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC9[i], PRE9[i], lw=0.7, label='Precision-Recall fold 9 (Aupr=%0.4f)' % aupr9)
    plt.legend()  #
# Precision-recall curve fold 10
for i in range(len(REC10)):
    aupr10 = metrics.auc(REC10[i], PRE10[i])
    print('AUPR=', aupr10)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC10[i], PRE10[i], lw=0.7, label='Precision-Recall fold 10 (Aupr=%0.4f)' % aupr10)
    plt.legend()  #
    #plt.legend()  #

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()