import torch as t
from torch import nn


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, target, prediction, drug_k, dis_k, drug_lap, dis_lap, alpha1, alpha2, sizes,weight_decay):
        loss_ls = t.norm((target - prediction), p='fro') ** 2

        drug_reg = t.trace(t.mm(t.mm(alpha1.T, drug_lap), alpha1))
        mic_reg = t.trace(t.mm(t.mm(alpha2.T, dis_lap), alpha2))
        graph_reg = sizes.lambda1 * drug_reg + sizes.lambda2 * mic_reg
        out1 = t.mm(drug_k, alpha1)
        out2 = t.mm(dis_k, alpha2)
        OUT=out1*out2.T
        #loss_ls =t.norm((target - OUT), p='fro') ** 2
        #loss_sum = loss_ls + graph_reg
        A1=t.norm(alpha1, p='fro') ** 2
        A2=t.norm(alpha2, p='fro') ** 2
        loss_sum =loss_ls + + weight_decay*(A1+A2)

        return loss_sum.sum()
