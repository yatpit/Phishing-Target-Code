# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 下午1:34
# @Author  : CSH_Yatpit

import torch
from torch import nn

from optUtils.pytorchModel import DeepLearningClassifier


class MLP_Clf(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', feat_len_list=()):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "dl_clf"
        self.feat_len_list = feat_len_list

    def create_model(self):
        embedding_dim = 32
        self.embedding_layer = nn.ModuleList([
            nn.Embedding(num_embeddings=f_size, embedding_dim=embedding_dim) for f_size in self.feat_len_list
        ])
        layer1 = 64
        self.mlp = nn.Sequential(
            nn.Linear(in_features=len(self.feat_len_list) * embedding_dim, out_features=layer1),
            nn.Tanh(),
            nn.Linear(in_features=layer1, out_features=self.label_num),
        )

    def forward(self, X):
        H = [emb_layer(x) for emb_layer, x in zip(self.embedding_layer, X.T)]
        y = torch.cat(H, dim=-1)
        y = self.mlp(y)
        return y
