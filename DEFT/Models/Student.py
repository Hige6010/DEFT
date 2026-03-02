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

class SimpleMLP(nn.Module):


    def __init__(self, d_list, d_model, class_num, dropout=0.3):
        super(SimpleMLP, self).__init__()

        self.view_num = len(d_list)
        self.d_model = d_model

        self.embeddinglayers = setEmbedingModel(d_list, d_model)

        hidden_dim = d_model * 2
        self.mlp = nn.Sequential(
            nn.Linear(d_model * self.view_num, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, class_num)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        embedded_views = []
        for v in range(self.view_num):
            embedded = self.embeddinglayers[v](x[v])
            embedded_views.append(embedded)
        concatenated = torch.cat(embedded_views, dim=1)  # [batch_size, d_model * view_num]

        features = self.mlp(concatenated)  # [batch_size, d_model]

        logits = self.classifier(features)  # [batch_size, class_num]

        return logits, features

class Model(nn.Module):

    def __init__(self, d_list,
                 d_model, n_layers, heads,
                 class_num, tau,
                 dropout,
                 ):
        super().__init__()
        self.class_num = class_num
        self.view_num = len(d_list)
        self.tau = tau
        self.mlp_model = SimpleMLP(d_list, d_model, class_num, dropout)

        self.Classifier = self.mlp_model.classifier

    def forward(self, x):
        return self.mlp_model(x)

def S_model(d_list,
            d_model=768,
            n_layers=2, heads=4, classes_num=10, tau=0.5, dropout=0.2,
            load_weights=None,
            device=torch.device('cuda:0')):
    simplified_n_layers = 1
    simplified_heads = 4
    simplified_dropout = 0.3
    if d_model % simplified_heads != 0:
        d_model = (d_model // simplified_heads) * simplified_heads

    assert d_model % simplified_heads == 0
    assert simplified_dropout < 1

    model = Model(d_list,
                  d_model=d_model,
                  n_layers=simplified_n_layers,
                  heads=simplified_heads,
                  class_num=classes_num,
                  tau=tau,
                  dropout=simplified_dropout)

    if load_weights is not None:
        print("loading pretrained weights...")
    else:

        pass

    model = model.to(device)

    return model