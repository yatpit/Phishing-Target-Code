# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 17:46
# @Author  : Esme_Chan

import pandas as pd
from sklearn.metrics import accuracy_score

from mlp_model import MLP_Clf
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_samples
from data_process import dataset
from optUtils.metricsUtil import f1_macro_score, f1_weighted_score

if __name__ == "__main__":
    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']

    feature_list = ["scheme", "domain", "top_domain", "second_domain", "domain_level", "domain_dash_count",
                    "domain_num_count", "slash_count", "special_symbol_count", "domain_len", "behind_domain_len",
                    "top_char", "top_special", "sens_words_url", "url_word_top3",
                    "valid_days", "registrant_country", "A", "A_1", "A_2", "A_IP_num", "CNAME",
                    "icon_str", "sens_words_html", "brand_words_html", "tfidf_top3", 'tag_count', 'html_text_symbol',
                    "brand_words_ocr", "sens_words_ocr"]
    label = 'target'

    for feat in feature_list:
        dataset[feat] = pd.factorize(dataset[feat])[0]
        dataset[feat] = pd.factorize(dataset[feat])[0]
    dataset[label] = pd.factorize(dataset[label])[0]

    feat_len_list = [dataset[feat].nunique() for feat in feature_list]

    X = dataset[feature_list].values
    y = dataset[label].values

    # 数据按折数分层排列
    X, y = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)

    train_point = int(len(X) * 0.2)

    # 训练
    model = MLP_Clf(learning_rate=0.001, epochs=100, batch_size=100, random_state=seed, feat_len_list=feat_len_list)
    model.param_search = False
    model.only_save_last_epoch = True
    model.metrics = accuracy_score
    model.metrics_list = [f1_macro_score, f1_weighted_score]
    model.fit(X[train_point:], y[train_point:], X[:train_point], y[:train_point])
