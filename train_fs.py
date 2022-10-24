# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 21:13
# @Author  : Esme_Chan
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_process import dataset
from fusion_model import FS_Clf_2
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_samples
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

    X1 = dataset[feature_list].values

    url_len = 100
    X_url = dataset['url'].apply(lambda x: np.fromstring(x.split('://')[-1], dtype=np.uint8)[:url_len]).values
    X2 = np.zeros((len(X_url), url_len)).astype('uint8')
    for i in range(len(X_url)):
        X2[i, :len(X_url[i])] = X_url[i]
    X2 = X2.astype('float') / 255

    y = dataset[label].values

    # 数据按折数分层排列
    X = [X1, X2]  # 融合一维数据和多维数据
    X, y = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)

    train_point = int(len(X) * 0.2) if type(X) != list else int(len(X[0]) * 0.2)

    # 融合分类器训练
    model = FS_Clf_2(learning_rate=0.001, epochs=200, batch_size=100, random_state=seed, feat_len_list=feat_len_list, url_len=url_len)
    model.model_name += '_common'
    model.param_search = False
    model.metrics = accuracy_score
    model.metrics_list = [f1_macro_score, f1_weighted_score]
    model.fit([x[train_point:] for x in X], y[train_point:], [x[:train_point] for x in X], y[:train_point])