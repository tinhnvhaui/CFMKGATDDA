import numpy as np
import pandas as pd
import math
import torch
import os
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

DiFeaMatrix=np.array(pd.read_csv("CDataset_DiseaseSim.csv", header=None))
print("Kich thuoc ma tran DiFeaMatrix la:",DiFeaMatrix.shape)

DrFeaMatrix=np.array(pd.read_csv("CDataset_DrugSim.csv", header=None))
print("Kich thuoc ma tran DrFeaMatrix la:",DrFeaMatrix.shape)

MM = np.array(DrFeaMatrix)
DD = np.array(DiFeaMatrix)
print("Kich thuoc ma tran DD la:", DD.shape)
print("ma tran DD la: ", DD)
print("Kich thuoc ma tran MM la:", MM.shape)
print("ma tran MM la: ", MM)

EIG = []# feature matrix of total sample
for i in range(DD.shape[0]):
    for j in range(MM.shape[0]):
        eig = np.hstack((DD[i],MM[j]))#feature vector length :DD.shape[0]+MM.shape[0]
        EIG.append(eig)
#  EIG[i][j] The eigenvector of the sample (d, m), and the corresponding label matrix is DM.
EIG_t = np.array(EIG).reshape(DD.shape[0],MM.shape[0],DD.shape[0]+MM.shape[0])
print("Ma tran feature vecture la: ", EIG_t)
#np.savetxt('matranfeaturevector.txt', EIG_t.astype(float))
EIG_t
print(EIG_t)
MD = pd.read_csv("CDataset_DiDrA.csv", header=None)
MD =MD .T
MD_c = MD.copy()
MD_c.columns=range(0,MD.shape[1])
MD_c.index=range(0,MD.shape[0])
MD_c=np.array(MD_c)
MD_c

print("ma tran MD_c la:", MD_c)

#Define random number seed
def setup_seed(seed):
    torch.manual_seed(seed)#
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)#
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False

DM_lable = MD_c.T
DM_lable
print('DM-lable:', DM_lable)

DM_lable = np.array(DM_lable)
lable = DM_lable.reshape(DM_lable.size).copy()  #
lable
print('lable:', lable)

PS_sub = np.where(lable==1)[0]#
PS_sub
print('PS_sub:', PS_sub)

PS_num = len(np.where(lable==1)[0])# Positive sample number
NS1 = np.where(lable==0)[0]#
# NS_sub = np.array(random.sample(list(NS1),PS_num))#
# NS_sub
NS1
print('NS1:', NS1)
NS1.shape
print('NS1.shape=', NS1.shape)

#Take the labels and feature vectors corresponding to all negative samples.
N_SAMPLE_lable = []#Label  corresponding to the sample
N_CHA=[]#The eigenvector matrix of the sample
for i in NS1:
    N_SAMPLE_lable.append(lable[i])
    N_CHA.append(EIG[i])
N_CHA = np.array(N_CHA)
print("N_SAMPLE_lable",N_SAMPLE_lable)
print("N_CHA",N_CHA)
print(np.array(N_CHA).shape)

kmeans=KMeans(n_clusters=23, random_state=36).fit(N_CHA)
kmeans
print('kmeans=', kmeans)

center=kmeans.cluster_centers_
center
labels=kmeans.labels_
labels
print('labels=', labels)
print('center.shape=', center.shape)
print('labels.shape',labels.shape)

type1=[]
type2=[]
type3=[]
type4=[]
type5=[]
type6=[]
type7=[]
type8=[]
type9=[]
type10=[]
type11=[]
type12=[]
type13=[]
type14=[]
type15=[]
type16=[]
type17=[]
type18=[]
type19=[]
type20=[]
type21=[]
type22=[]
type23=[]
for i in range(len(labels)):
    if labels[i]==0:
        type1.append(NS1[i])
    if labels[i]==1:
        type2.append(NS1[i])
    if labels[i]==2:
        type3.append(NS1[i])
    if labels[i]==3:
        type4.append(NS1[i])
    if labels[i]==4:
        type5.append(NS1[i])
    if labels[i]==5:
        type6.append(NS1[i])
    if labels[i]==6:
        type7.append(NS1[i])
    if labels[i]==7:
        type8.append(NS1[i])
    if labels[i]==8:
        type9.append(NS1[i])
    if labels[i]==9:
        type10.append(NS1[i])
    if labels[i]==10:
        type11.append(NS1[i])
    if labels[i]==11:
        type12.append(NS1[i])
    if labels[i]==12:
        type13.append(NS1[i])
    if labels[i]==13:
        type14.append(NS1[i])
    if labels[i]==14:
        type15.append(NS1[i])
    if labels[i]==15:
        type16.append(NS1[i])
    if labels[i]==16:
        type17.append(NS1[i])
    if labels[i]==17:
        type18.append(NS1[i])
    if labels[i]==18:
        type19.append(NS1[i])
    if labels[i]==19:
        type20.append(NS1[i])
    if labels[i]==20:
        type21.append(NS1[i])
    if labels[i]==21:
        type22.append(NS1[i])
    if labels[i]==22:
        type23.append(NS1[i])

type23
print('len(type23)=', len(type23))
print('len(type22)=', len(type22))
print('len(type21)=', len(type21))
print('len(type20)=', len(type20))
print('len(type19)=', len(type19))
print('len(type18)=', len(type18))
print('len(type17)=', len(type17))
print('len(type16)=', len(type16))
print('len(type15)=', len(type15))
print('len(type14)=', len(type14))
print('len(type13)=', len(type13))
print('len(type12)=', len(type12))
print('len(type11)=', len(type11))
print('len(type10)=', len(type10))
print('len(type09)=', len(type9))
print('len(type08)=', len(type8))
print('len(type07)=', len(type7))
print('len(type06)=', len(type6))
print('len(type05)=', len(type5))
print('len(type04)=', len(type4))
print('len(type03)=', len(type3))
print('len(type02)=', len(type2))
print('len(type01)=', len(type1))

type=[type1,type2,type3,type4,type5,type6,type7,type8,type9,type10,type11,type12,type13,
      type14,type15,type16,type17,type18,type19,type20,type21,type22,type23]
print('type=', type)
setup_seed(36)

# Select a negative sample equal to the positive sample.
type=[type1,type2,type3,type4,type5,type6,type7,type8,type9,type10,type11,type12,type13,
      type14,type15,type16,type17,type18,type19,type20,type21,type22,type23]
mtype=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for k in range(23):
    mtype[k]=random.sample(type[k],196)
mtype
print('len(mtype)=', len(mtype))

# Negative sample subscript
NS_sub=[]
for i in range(len(lable)):
    for z2 in range(23):
        if i in mtype[z2]:
            NS_sub.append(i)
NS_sub
print('len(NS_sub=',len(NS_sub))
SAMPLE_sub = np.hstack((PS_sub,NS_sub))#
print('SAMPLE_sub',SAMPLE_sub)

#The label and feature vector corresponding to this sample
SAMPLE_lable = []#Labels 0 and 1 corresponding to samples
CHA=[]#The eigenvector matrix of the sample
for i in SAMPLE_sub:
    SAMPLE_lable.append(lable[i])
    CHA.append(EIG[i])
CHA = np.array(CHA)
print("SAMPLE_lable",SAMPLE_lable)
print("CHA.SHAPE=", CHA.shape)
print("CHA",CHA)

# Define some global constants
BETA = math.pow(10,-7)
N_INP = CHA.shape[1]#Input dimension
N_HIDDEN = 576#Hide layer dimension
N_EPOCHS = 100# Epoch times
use_sparse = True #Sparse or not


class DSAE(nn.Module):
    def __init__(self):
        super(DSAE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=N_INP, out_features=N_HIDDEN),
            nn.Sigmoid(),
            nn.Linear(N_HIDDEN, int(N_HIDDEN / 2)),
            nn.Sigmoid(),
            nn.Linear(int(N_HIDDEN / 2), int(N_HIDDEN / 4)),
            nn.Sigmoid()
            # nn.Linear(int(N_HIDDEN / 4), int(N_HIDDEN / 8)),
            # nn.Sigmoid()
        )
        # decoder
        self.decoder = nn.Sequential(
            # nn.Linear(int(N_HIDDEN / 8), int(N_HIDDEN / 4)),
            # nn.Sigmoid(),
            nn.Linear(int(N_HIDDEN / 4), int(N_HIDDEN / 2)),
            nn.Sigmoid(),
            nn.Linear(int(N_HIDDEN / 2), N_HIDDEN),
            nn.Sigmoid(),
            nn.Linear(N_HIDDEN, N_INP),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

net = DSAE()
print('net=',net)

# Transform feature vectors into tensors
# T = np.array(T)
CHA_t = CHA.copy()
CHA_t = torch.from_numpy(CHA_t)
# CHA_t = torch.as_tensor(CHA_t)
# CHA = CHA.float()
CHA_t = CHA_t.float()
CHA_t
print('CHA_t.shape=', CHA_t.shape)

# Define and save network functions.
def save(net, path):
    torch.save(net.state_dict(), path)


# Define training network function, network, loss evaluation, optimizer and training set.
def train(net, trainloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_rate = []
    lr_t = []  # 存储学习率

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # Network parameters, learning rate
    #     scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,70,90], gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(N_EPOCHS):  #
        optimizer.zero_grad()  #

        # forward + backward + optimize
        #         encoded = net.encode(trainloader)
        encoded, decoded = net(trainloader)  #
        #         print("encoded",encoded)
        #         print("decoded",decoded)

        # loss
        inputs1 = trainloader.view(trainloader.size(0), -1)
        loss1 = criterion(decoded, inputs1)
        #             print(loss1)

        if use_sparse:  #
            kl_loss = torch.sum(torch.sum(trainloader * (torch.log(trainloader / decoded)) + (1 - trainloader) * (
                torch.log((1 - trainloader) / (1 - decoded)))))
            #             p=trainloader
            #             q=decoded
            #             kl_loss = F.kl_div(q.log(),p,reduction="sum")+F.kl_div((1-q).log(),1-p,reduction="sum")
            loss = loss1 + BETA * kl_loss
        #             print(kl_loss)
        #             print(loss1)
        else:
            loss = loss1
        loss.backward()  #
        optimizer.step()  #

        #
        scheduler.step(loss)
        print("[%d] loss: %.5f" % (epoch + 1, loss))
        lr = optimizer.param_groups[0]['lr']
        lr_t.append(lr)
        print("epoch={}, lr={}".format(epoch + 1, lr_t[-1]))
        loss_t = loss.clone()
        loss_rate.append(loss_t.cpu().detach().numpy())
    x = list(range(len(lr_t)))
    plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=None, wspace=0.5)
    plt.subplot(121)
    plt.title('(a)', x=-0.2, y=1)
    #     loss_rate = ['{:.5f}'.format(i) for i in loss_rate]
    plt.plot(x, loss_rate, label='loss change curve', color="#f59164")
    plt.ylabel('loss changes')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(122)
    plt.title('(b)', x=-0.2, y=1)
    plt.plot(x, lr_t, label='lr curve', color="#f59164")
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend()
    # plt.savefig("E:/loss_lr.png",dpi=300)
    plt.show()
    print('Finished Training')

path = 'E:/Data/DSAE_RF5.pth'
save(net, path)
net.load_state_dict(torch.load('E:/Data/DSAE_RF5.pth'))
torch.load('E:/Data/DSAE_RF5.pth')

# predict
net.eval()    #test
sample_extract=net(CHA_t)
sample_extract
print('sample_extract', sample_extract)
#np.savetxt('FinalPredictedResults_Coriginal_04082024.txt', sample_extract.detach().numpy())
#print('sample_extract shape=',sample_extract.shape)
encoded = sample_extract[0]
decoded = sample_extract[1]
#sample_extract1=sample_extract.detach().numpy()
np.savetxt('FinalResults_decoded_Coriginal_04062024.txt', decoded.detach().numpy())
# Reserved features
SAMPLE_feature = encoded.detach().numpy()#
SAMPLE_feature
print('SAMPLE_feature', SAMPLE_feature.shape)
np.savetxt('SampleFeature_all_Coriginal_04082024.txt', SAMPLE_feature)
SAMPLE_lable = np.array(SAMPLE_lable)# Sample label
SAMPLE_lable
print('SAMPLE_lable', SAMPLE_lable)
#np.savetxt('PredictedResults_all_Coriginal_04062024_new.txt', SAMPLE_lable)
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

y_real = []
y_proba = []

kf = KFold(n_splits=10, shuffle=True, random_state=36)
i=0
for train_index, test_index in kf.split(SAMPLE_sub):
    i=i+1
    train_features = SAMPLE_feature[train_index]
    test_features = SAMPLE_feature[test_index]
    train_labels = SAMPLE_lable[train_index]
    test_labels = SAMPLE_lable[test_index]
    # normalize
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    #     print(SAMPLE_sub[train_index])
    rf = RandomForestClassifier(n_estimators=1100, criterion='entropy', max_depth=23, bootstrap=False,
                                min_samples_leaf=5, min_samples_split=6,
                                max_features='sqrt', random_state=36).fit(train_features,
                                                                          train_labels)  # Enter the training set and the label of the training set.

    #     test_score1 = clf1.score(test_features, test_labels)#ACC
    test_predict = rf.predict(test_features)
    #np.savetxt('YTestPredict_CDataset_DrDi_FTest%d_04062024.txt'%i, test_predict)
    pre = rf.predict_proba(test_features)[:, 1]
    #np.savetxt('YProPredict1_CDataset_DrDi_FPro%d_04062024.txt'%i, pre)
    tru = test_labels
    pre2 = test_predict

    # auc
    auc = roc_auc_score(tru, pre)
    fpr, tpr, thresholds1 = metrics.roc_curve(tru, pre,
                                              pos_label=1)  # The actual value indicated as 1 is a positive sample.
    FPR.append(fpr)
    TPR.append(tpr)
    THR1.append(thresholds1)
    AUC.append(auc)
    print("auc:", auc)
    # ACC
    acc = accuracy_score(tru, pre2)
    ACC.append(acc)
    print("acc:", acc)
    # aupr
    precision, recall, thresholds2 = precision_recall_curve(tru, pre)
    aupr = metrics.auc(recall, precision)  #
    AUPR.append(aupr)
    THR2.append(thresholds2)
    PRE.append(precision)
    REC.append(recall)
    y_real.append(test_labels)
    y_proba.append(pre)
    print("aupr:", aupr)
    # recall
    recall1 = metrics.recall_score(tru, pre2, average='macro')  #
    RECALL.append(recall1)
    print("recall:", recall1)
    # precision
    precision1 = metrics.precision_score(tru, pre2, average='macro')
    print("precision:", precision1)
    PREECISION.append(precision1)
    # f1_score
    f1 = metrics.f1_score(tru, pre2, average='macro')  # F1
    F1.append(f1)
    print("f1:", f1)

np.savetxt('YPredictFinal_CDataset_DrDi_Matrix_Original_04082024.txt', y_proba)
np.savetxt('YReal_CDataset_DrDi_Matrix_Original_04082024.txt', y_real)
print("AUC:", np.average(AUC))
print("ACC:", np.average(ACC))
print("AUPR:", np.average(AUPR))
print("RECALL:", np.average(RECALL))
print("PREECISION:", np.average(PREECISION))
print("F1:", np.average(F1))

#np.savetxt('Y_Real_K2r05_new.txt', y_real)
#np.savetxt('Y_proba_K2r05_new.txt',y_proba)
# ROC curve
plt.figure()
tprs=[]
mean_fpr=np.linspace(0,1,1000)
for i in range(len(FPR)):
    tprs.append(np.interp(mean_fpr,FPR[i],TPR[i]))
    tprs[-1][0]=0.0
    auc = metrics.auc(FPR[i], TPR[i])
#     print(auc)
#     plt.xlim(0, 1)  #
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(FPR[i], TPR[i],lw=1,label='ROC fold %d (auc=%0.4f)' % (i, auc))
    plt.legend()  #
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
# plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Base Line')  #
mean_auc = metrics.auc(mean_fpr, mean_tpr)  #
std_auc = np.std(tprs, axis=0)
plt.plot(mean_fpr,mean_tpr,color="#D81C38",lw=2,label='Mean ROC (mean auc=%0.4f)' % (np.average(AUC)))
#
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#
plt.title("Receiver Operating Characteristic Curve")
plt.legend(loc="lower right")
#plt.legend(bbox_to_anchor = (1.05, 0), loc=3, borderaxespad = 0)#
# plt.legend()

plt.show()


# In[67]:


# PR curve
plt.figure()
for i in range(len(REC)):
    aupr = metrics.auc(REC[i], PRE[i])
#     print(auc)
#     plt.xlim(0, 1)  #
#     plt.ylim(0, 1)  #
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(REC[i], PRE[i],lw=1,label='PR fold %d (aupr=%0.4f)' % (i, aupr))
    plt.legend()  #

# plt.plot([0,1],[1,0],linestyle='--',lw=2,color='r',label='Base Line')  #

y_real1 = np.concatenate(y_real)
np.savetxt('YReal_CDataset_DrDi_Original_Matrix.txt', y_real1)
y_proba1 = np.concatenate(y_proba)
np.savetxt('YProba_CDataset_DrDi_Original_Matrix.txt',y_proba1)
precisions, recalls, _ = precision_recall_curve(y_real1, y_proba1)


plt.plot(recalls, precisions, lw=2,color="#D81C38", label='Mean PR (mean aupr=%0.4f)' % (np.average(AUPR)))
#
plt.xlabel("Recall")
plt.ylabel("Precision")
#
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
#plt.legend(bbox_to_anchor = (1.05, 0), loc=3, borderaxespad = 0)#
# plt.legend()

plt.show()
# np.savetxt('Y_Real1_K2R5.txt', y_real1)
# np.savetxt('Y_Proba1_K2R5.txt', y_proba1)