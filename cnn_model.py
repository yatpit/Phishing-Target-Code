#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/12/17 15:19
# @Author : LYX-夜光
from torch import nn

from optUtils.pytorchModel import DeepLearningClassifier


def get_conv_dim(in_dim, kernel, stride, max_size=1):
    return int(int((in_dim - kernel + stride) / stride) / max_size)


class CNN_Clf(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', re_size=64):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "cnn_clf"
        self.re_size = re_size

    def create_model(self):
        conv1_size = 16
        conv1_kernel, conv1_stride = 4, 4
        pool1_dim = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv1_size, kernel_size=conv1_kernel, stride=conv1_stride),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=pool1_dim, stride=pool1_dim),
        )

        conv1_out_dim = get_conv_dim(self.re_size, conv1_kernel, conv1_stride, pool1_dim)

        print(conv1_size * conv1_out_dim * conv1_out_dim)

        out_1_dim = 512
        out_2_dim = 128
        self.out_layer = nn.Sequential(
            nn.Linear(in_features=conv1_size * conv1_out_dim * conv1_out_dim, out_features=out_1_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_1_dim, out_features=out_2_dim),
            nn.Tanh(),
            nn.Linear(in_features=out_2_dim, out_features=self.label_num)
        )

    def forward(self, X):
        H = self.conv1(X.permute(0, 3, 1, 2))
        y = H.flatten(1)
        y = self.out_layer(y)
        return y