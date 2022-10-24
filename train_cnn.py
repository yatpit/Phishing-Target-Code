#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/12/17 15:19
# @Author : LYX-夜光

import pandas as pd
from sklearn.metrics import accuracy_score

from cnn_model import CNN_Clf
from data_process import dataset, picture_data
from fusion_model import FS_Clf
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_samples
from optUtils.metricsUtil import f1_macro_score, f1_weighted_score

if __name__ == "__main__":
    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']

    label = 'target'
    dataset[label] = pd.factorize(dataset[label])[0]

    pic_type = 'picture'
    re_size = 64
    X = picture_data(dataset, pic_type, re_size=re_size)
    y = dataset[label].values

    # 数据按折数分层排列
    X, y  = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)

    train_point = int(len(X) * 0.2)

    # 融合分类器训练
    model = CNN_Clf(learning_rate=0.001, epochs=100, batch_size=100, random_state=seed, re_size=re_size)
    model.param_search = False
    model.only_save_last_epoch = True
    model.metrics = accuracy_score
    model.metrics_list = [f1_macro_score, f1_weighted_score]
    model.fit(X[train_point:], y[train_point:], X[:train_point], y[:train_point])