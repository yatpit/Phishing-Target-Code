#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/12/20 22:00
# @Author : LYX-夜光

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_samples
from data_process import dataset
from optUtils.metricsUtil import f1_macro_score, f1_weighted_score
from optUtils.modelUtil import model_registration, model_selection
from optUtils.trainUtil import ml_train

if __name__ == "__main__":
    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']

    feature_list = ["scheme", "domain", "top_domain", "second_domain", "domain_level", "domain_dash_count",
                    "domain_num_count", "slash_count", "special_symbol_count", "domain_len", "behind_domain_len",
                    "top_char", "top_special", "sens_words_url", "url_word_top3",
                    "valid_days", "registrant_country", "A", "A_1", "A_2", "A_IP_num", "CNAME",
                    "icon_str", "sens_words_html", "brand_words_html", "tfidf_top3", 'tag_count', 'html_text_symbol',
                    "brand_words_ocr", "sens_words_ocr"]
    label = 'target'

    # dataset = pd.read_csv('dataset_result.csv', index_col=None, encoding='utf-8')

    for feat in feature_list:
        dataset[feat] = pd.factorize(dataset[feat])[0]
        dataset[feat] = pd.factorize(dataset[feat])[0]
    dataset[label] = pd.factorize(dataset[label])[0]

    X = dataset[feature_list].values
    y = dataset[label].values
    # print(X[:10])
    # exit()
    # 数据按折数分层排列
    X, y = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)

    train_point = int(len(X) * 0.2)

    model_registration(
        lgb_clf=LGBMClassifier,
        xgb_clf=XGBClassifier,
    )
    model_name_list = ['lr_clf', 'dt_clf', 'svm_clf', 'knn_clf', 'rf_clf', 'lgb_clf', 'xgb_clf']
    metrics_list = [accuracy_score, f1_macro_score, f1_weighted_score]  # 添加多个评价指标
    for model_name in model_name_list:
        model_param = {} if model_name == "knn_clf" else {"random_state": seed}
        # 机器学习常规训练
        ml_train(X[train_point:], y[train_point:], X[:train_point], y[:train_point], model_name, model_param,
                 metrics_list)

    # for model_name in model_name_list:
    #     if model_name == "knn_clf":
    #         param = {}
    #     else:
    #         param = {"random_state": seed}
    #     model = model_selection(model_name, **param)
    #     model.fit(X[train_point:], y[train_point:])
    #     y_train_pred = model.predict(X[train_point:])
    #     print("%s - train_acc: %.6f - train_f1_macro: %6f - train_f1_weighted: %6f" % (
    #         model_name,
    #         accuracy_score(y[train_point:], y_train_pred),
    #         f1_macro_score(y[train_point:], y_train_pred),
    #         f1_weighted_score(y[train_point:], y_train_pred),
    #     ))
    #     y_val_pred = model.predict(X[:train_point])
    #     print("%s - val_acc: %.6f - val_f1_macro: %6f - val_f1_weighted: %6f" % (
    #         model_name,
    #         accuracy_score(y[:train_point], y_val_pred),
    #         f1_macro_score(y[:train_point], y_val_pred),
    #         f1_weighted_score(y[:train_point], y_val_pred),
    #     ))
