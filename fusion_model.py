# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 下午1:47
# @Author  : CSH_Yatpit
import numpy as np
import torch
from torch import nn

from optUtils.pytorchModel import DeepLearningClassifier


def get_conv_dim(in_dim, kernel, stride, max_size=1):
    return int(int((in_dim - kernel + stride) / stride) / max_size)


class FS_Clf(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', feat_len_list=(),
                 re_size=64):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "fs_clf"
        self.feat_len_list = feat_len_list
        self.re_size = re_size

    def create_model(self):
        embedding_dim = 16
        self.embedding_layer = nn.ModuleList([
            nn.Embedding(num_embeddings=f_size, embedding_dim=embedding_dim) for f_size in self.feat_len_list
        ])
        mlp1_dim = 32
        self.mlp = nn.Sequential(
            nn.Linear(in_features=len(self.feat_len_list) * embedding_dim, out_features=mlp1_dim),
        )

        conv1_size = 16
        conv1_kernel, conv1_stride = 4, 4
        pool1_dim = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_size, kernel_size=conv1_kernel, stride=conv1_stride),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=pool1_dim, stride=pool1_dim),
        )
        conv1_out_dim = get_conv_dim(self.re_size, conv1_kernel, conv1_stride, pool1_dim)
        out_1_dim = 512
        out_2_dim = 128
        self.conv_out = nn.Sequential(
            nn.Linear(in_features=conv1_size * conv1_out_dim * conv1_out_dim, out_features=out_1_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_1_dim, out_features=out_2_dim),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(in_features=mlp1_dim + out_2_dim, out_features=self.label_num),
        )

    def forward(self, X):
        X1, X2 = X

        H1 = [emb_layer(x1) for emb_layer, x1 in zip(self.embedding_layer, X1.T)]
        y1 = self.mlp(torch.cat(H1, dim=-1))

        H2 = self.conv1(X2.permute(0, 3, 1, 2))
        y2 = H2.flatten(1)
        y2 = self.conv_out(y2)

        y = torch.cat([y1, y2], dim=-1)
        y = self.out_layer(y)
        return y


class FS_Clf_2(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', feat_len_list=(),
                 url_len=100):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "fs_clf_2"
        self.feat_len_list = feat_len_list
        self.url_len = url_len

    def create_model(self):
        embedding_dim = 32
        self.embedding_layer = nn.ModuleList([
            nn.Embedding(num_embeddings=f_size, embedding_dim=embedding_dim) for f_size in self.feat_len_list
        ])
        mlp1_dim = 512
        mlp2_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(in_features=len(self.feat_len_list) * embedding_dim, out_features=mlp1_dim),
            nn.Tanh(),
            nn.Linear(in_features=mlp1_dim, out_features=mlp2_dim),
        )
        conv1_size = 16
        conv1_kernel, conv1_stride = 5, 5
        pool1_dim = 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, conv1_size, kernel_size=conv1_kernel, stride=conv1_stride),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=pool1_dim, stride=pool1_dim),
        )

        conv1_out_dim = get_conv_dim(self.url_len, conv1_kernel, conv1_stride, pool1_dim)

        print(mlp2_dim, conv1_size * conv1_out_dim)

        out_1_dim = 512
        out_2_dim = 64
        self.out_layer = nn.Sequential(
            nn.Linear(in_features=mlp2_dim + conv1_size * conv1_out_dim, out_features=out_1_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_1_dim, out_features=out_2_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_2_dim, out_features=self.label_num)
        )

    def forward(self, X):
        X1, X2 = X
        H1 = [emb_layer(x1) for emb_layer, x1 in zip(self.embedding_layer, X1.T)]
        y1 = self.mlp(torch.cat(H1, dim=-1))
        H2 = self.conv1(X2.unsqueeze(1))
        y2 = H2.flatten(1)
        y = torch.cat([y1, y2], dim=-1)
        y = self.out_layer(y)
        return y
