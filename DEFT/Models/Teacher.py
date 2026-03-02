# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/3/30
import torch
import torch.nn as nn

import copy
import math
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def setEmbedingModel(d_list, d_out):
    return nn.ModuleList([nn.Linear(d, d_out) for d in d_list])


class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class crs_attention(nn.Module):
    def __init__(self, in_dim, label_embedd):
        super(crs_attention, self).__init__()
        self.layerQ = nn.Linear(label_embedd, in_dim)
        self.layerK = nn.Linear(in_dim, in_dim)
        self.layerV = nn.Linear(in_dim, in_dim)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(p=0.1)
        self.d_k = in_dim
        self.initialize()

    def initialize(self):
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()
        self.layerV.reset_parameters()

    def forward(self, node_emb, label_emb, tau=0.5):
        Q = self.layerQ(label_emb)  # [128, 512]
        K = self.layerK(node_emb)
        V = self.layerV(node_emb)  # [128, 3, 512]

        attn_1 = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [128, 1, 3]
        mask = torch.tril(torch.ones_like(attn_1))
        attn_1_score = attn_1.masked_fill(mask == 0, -1e9)

        attn_1_weight = F.softmax(attn_1_score * tau, dim=-1)
        attn_1_weight = self.proj_drop(attn_1_weight)

        attn_2 = torch.matmul(K, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask_2 = torch.tril(torch.ones_like(attn_2))
        attn_2_score = attn_2.masked_fill(mask_2 == 0, -1e9)
        attn_2_weight = F.softmax(attn_2_score * tau, dim=-1)
        attn_2_weight = self.proj_drop(attn_2_weight)
        attn_2_new = torch.matmul(attn_2_weight, V)

        attn = attn_1_weight @ V
        attn = self.proj(attn)

        return attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, label_embedd, heads, view_num, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.label_attentive = crs_attention(d_model, label_embedd)
        self.ff = FeedForward(d_model, dropout=0.2)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, gt, tau):
        x_norm = self.norm_1(x)  # [bs, view, dim]
        lb_mv = self.dropout_1(self.label_attentive(x, gt, tau))
        x = x_norm + lb_mv
        #x= x_norm
        x2 = self.norm_2(x)
        x_new = x + self.dropout_2(self.ff(x2))

        return x_new


class Encoder(nn.Module):
    def __init__(self, d_model, label_embedd, N, heads, dropout, view_num):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, label_embedd, heads, view_num, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, gt, tau):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, gt, tau)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, label_embedd, N, heads, dropout, view_num):
        super().__init__()
        self.encoder = Encoder(d_model, label_embedd, N, heads, dropout, view_num)

    def forward(self, src, gt, tau):
        e_outputs = self.encoder(src, gt, tau)
        return e_outputs


class Classifier(nn.Module):
    def __init__(self, nhid, nclass, dropout=0., with_bn=True, with_bias=True):
        super(Classifier, self).__init__()
        self.with_bn = with_bn
        self.layer1 = nn.Linear(nhid, int(nhid / 2), bias=with_bias)
        self.layer2 = nn.Linear(int(nhid / 2), nclass, bias=with_bias)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(int(nhid / 2))

        self.dropout = dropout

        self.initialize()

    def initialize(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.bn1.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        return x


class LabelEncoder(nn.Module):
    def __init__(self, nhid, nclass, nlayers=3, dropout=0.5, with_bn=True, with_bias=True):
        super(LabelEncoder, self).__init__()
        self.with_bn = with_bn
        self.layers = nn.ModuleList()
        self.dropout = dropout

        self.layers.append(nn.Linear(nclass, nhid, bias=with_bias))
        if with_bn:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(nhid))

        for i in range(nlayers - 2):
            self.layers.append(nn.Linear(nhid, nhid, bias=with_bias))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, nhid, bias=with_bias))

        self.initialize()

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for m in self.bns:
                m.reset_parameters()

    def forward(self, y):
        for ii, layer in enumerate(self.layers):
            if ii == len(self.layers) - 1:
                return layer(y)
            y = layer(y)
            if self.with_bn:
                y = self.bns[ii](y)

            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)


class Model(nn.Module):
    def __init__(self, d_list, label_embedd,
                 d_model, n_layers, heads,
                 class_num, tau,
                 dropout,
                 ):
        super().__init__()
        self.class_num = class_num
        self.d_model = d_model
        self.view_num = len(d_list)
        self.ETrans = Transformer(d_model, label_embedd, n_layers, heads, dropout, self.view_num)
        self.embeddinglayers = setEmbedingModel(d_list, d_model)  # embedding
        self.Classifier = Classifier(d_model, class_num, dropout=0.)
        self.LabelEncoder = LabelEncoder(label_embedd, class_num)
        self.tau = tau

    def forward(self, x, gt):
        for v in range(self.view_num):  # encode input view to features with same dimension
            x[v] = self.embeddinglayers[v](x[v])

        x = torch.stack(x, dim=1)  # B,view,dim

        gtEmbed = self.LabelEncoder(gt)  # Bs, d_model
        x = self.ETrans(x, gtEmbed, self.tau)

        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        EncX = x

        output = self.Classifier(x)

        return output, EncX, gtEmbed


def T_model(d_list,
            label_embedd=64,
            d_model=768,
            n_layers=2, heads=4, classes_num=10, tau=0.5, dropout=0.2,
            load_weights=None,
            device=torch.device('cuda:0')):
    """
    params: d_list-->list-->dims of view
    d_model--int--num of neurons
    """
    assert d_model % heads == 0
    assert dropout < 1

    model = Model(d_list, label_embedd, d_model, n_layers, heads, classes_num, tau, dropout)

    if load_weights is not None:
        print("loading pretrained weights...")
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model = model.to(device)

    return model