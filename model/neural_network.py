from itertools import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class fmmo_survival_Analysis(nn.Module):

    def __init__(self, p=0):
        nn.Module.__init__(self)
        self.miRNA_encoding = nn.Sequential(nn.Linear(1881, 1000),
                                            nn.Tanh(),
                                            nn.BatchNorm1d(1000),
                                            nn.Linear(1000, 500),
                                            nn.Tanh(),
                                            nn.BatchNorm1d(500),
                                            nn.Linear(500, 500),
                                            nn.Sigmoid())

        self.RNASeq_encoding = nn.Sequential(nn.Linear(17720, 1000),
                                             nn.Tanh(),
                                             nn.BatchNorm1d(1000),
                                             nn.Linear(1000, 500),
                                             nn.Tanh(),
                                             nn.BatchNorm1d(500),
                                             nn.Linear(500, 500),
                                             nn.Sigmoid())

        self.Wq_1 = nn.Linear(500, 200)
        self.Wk_1 = nn.Linear(500, 200)
        self.Wv_1 = nn.Linear(500, 200)

        self.Wq_2 = nn.Linear(500, 200)
        self.Wk_2 = nn.Linear(500, 200)
        self.Wv_2 = nn.Linear(500, 200)

        self.BN_RNASeq = nn.BatchNorm1d(200)
        self.BN_miRNA = nn.BatchNorm1d(200)

        self.atten_fuse_weight = nn.Linear(400, 40)
        self.BN = nn.BatchNorm1d(80)
        self.p = p
        self.bm_full = nn.Sequential(nn.Linear(200, 100, bias=False), nn.ReLU(), nn.Linear(100, 60, bias=False),
                                     nn.ReLU())
        self.hitting_layer = nn.Linear(60, 30, bias=False)
        self.softmax = nn.Softmax()

    def self_attention(self, gene, miRNA):
        # x: [batch_size, seq_len, input_dim]
        gene = torch.squeeze(gene)
        miRNA = torch.squeeze(miRNA)

        gene_1 = gene.unsqueeze(1)
        miRNA_1 = miRNA.unsqueeze(1)
        x = torch.cat((gene_1, miRNA_1), 1)
        print(x.shape)
        batch_size, seq_len, _ = x.size()
        Q = self.Wq_1(x)  # [batch_size, seq_len, hidden_dim]
        K = self.Wk_1(x)  # [batch_size, seq_len, hidden_dim]
        V = self.Wv_1(x)  # [batch_size, seq_len, hidden_dim]

        # 求注意力分数
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / (200 ** 0.5)
        # 对注意力分数归一化，最后一个维度做
        attention_weights = nn.functional.softmax(attention_weights, dim=-1)
        # 将归一化后的权重与value相乘
        attended_values = torch.matmul(attention_weights, V)
        print(attended_values)

        gene_2 = gene.unsqueeze(1)
        miRNA_2 = miRNA.unsqueeze(1)
        x = torch.cat((gene_2, miRNA_2), 1)

        Q_2 = self.Wq_2(x)  # [batch_size, seq_len, hidden_dim]
        K_2 = self.Wk_2(x)  # [batch_size, seq_len, hidden_dim]
        V_2 = self.Wv_2(x)  # [batch_size, seq_len, hidden_dim]

        # 求注意力分数
        attention_weights_2 = torch.matmul(Q_2, K_2.transpose(-2, -1)) / (200 ** 0.5)
        # 对注意力分数归一化，最后一个维度做
        attention_weights_2 = nn.functional.softmax(attention_weights_2, dim=-1)
        # 将归一化后的权重与value相乘
        attended_values_2 = torch.matmul(attention_weights_2, V_2)
        temp_1 = self.get_scale_feature(self.atten_fuse_weight(
            torch.cat((torch.squeeze(attended_values[:, 0, :]), torch.squeeze(attended_values[:, 1, :])), 1)))
        temp_2 = self.get_scale_feature(self.atten_fuse_weight(
            torch.cat((torch.squeeze(attended_values_2[:, 0, :]), torch.squeeze(attended_values_2[:, 1, :])), 1)))

        return torch.cat((self.BN(torch.squeeze(torch.cat((F.normalize(temp_1), F.normalize(temp_2)), 1))),
                          torch.zeros([gene.shape[0], 120])), 1)

    def get_scale_feature(self, temp_fea):
        return torch.sqrt(F.relu(temp_fea)) - torch.sqrt(F.relu(-temp_fea))

    def get_RNASeq_feature(self, gene):
        gene = gene.view(gene.shape[0], -1)
        # gene = F.tanh(self.fcg(gene))
        # gene = self.bn1_fcg(gene)
        # gene = self.fcg_highway(gene)
        # # gene = F.dropout(gene, 0.3)
        # # gene =F.sigmoid(self.bn2_fcg(gene))
        # return gene
        # gene_encoding = F.tanh(self.RNASeq_BN(self.RNASeq_encoding(gene)))
        gene_encoding = F.dropout(self.RNASeq_encoding(gene), self.p)
        return gene_encoding

    def get_miRNA_feature(self, miRNA):
        miRNA = miRNA.view(miRNA.shape[0], -1)
        # microRNA = F.tanh(self.fcm(microRNA))
        # microRNA = self.bn1_fcm(microRNA)
        # microRNA = self.fcm_highway(microRNA)
        # # microRNA  = F.dropout(microRNA, 0.3)
        # # microRNA_feature =F.sigmoid(self.bn2_fcm(microRNA))
        # return microRNA
        # miRNA_encoding = F.tanh(self.miRNA_BN(self.miRNA_encoding(miRNA)))
        miRNA_encoding = self.miRNA_encoding(miRNA)
        return miRNA_encoding

    def get_survival_result(self, gene, miRNA):
        return self.softmax(self.hitting_layer(self.get_survival_feature(gene, miRNA)))

    def get_survival_feature(self, gene, miRNA):
        RNASeq_feature = self.get_RNASeq_feature(gene)
        miRNA_feature = self.get_miRNA_feature(miRNA)
        mul_head_feature = self.self_attention(RNASeq_feature, miRNA_feature)
        return self.bm_full(mul_head_feature)
