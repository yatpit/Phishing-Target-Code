# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 下午1:46
# @Author  : CSH_Yatpit
import pandas as pd
from sklearn.metrics import accuracy_score

from data_process import dataset, picture_data
from fusion_model import FS_Clf
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

    # feature_list = ["scheme", "domain", "top_domain", "second_domain", "domain_level", "domain_dash_count",
    #                 "domain_num_count", "slash_count", "special_symbol_count", "domain_len", "behind_domain_len",
    #                 "top_char", "top_special", "url_word_top3", "sens_words_url"]

    # feature_list = ["valid_days", "registrant_country", "A", "A_1", "A_2", "A_IP_num", "CNAME"]
    # feature_list = ["icon_str", "sens_words_html", "tfidf_top3", "brand_words_html", "html_text_symbol", 'tag_count',
    #                 "brand_words_ocr", "sens_words_ocr"]

    # feature_list = ['top_special', 'sens_words_url', 'special_symbol_count', 'top_domain',
    #                 'icon_str', 'CNAME', 'domain_dash_count', 'domain_len', 'tfidf_top3',
    #                 'brand_words_html', 'sens_words_ocr', 'domain_num_count',
    #                 'html_text_symbol', 'tag_count', 'A_2', 'scheme', 'registrant_country',
    #                 'valid_days', 'url_word_top3', 'domain', 'A_1', 'domain_level', 'A',
    #                 'second_domain', 'brand_words_ocr']
    label = 'target'

    for feat in feature_list:
        dataset[feat] = pd.factorize(dataset[feat])[0]
        dataset[feat] = pd.factorize(dataset[feat])[0]
    dataset[label] = pd.factorize(dataset[label])[0]

    feat_len_list = [dataset[feat].nunique() for feat in feature_list]

    X1 = dataset[feature_list].values
    pic_type = 'picture'
    re_size = 64
    X2 = picture_data(dataset, pic_type, re_size=re_size)
    y = dataset[label].values

    # 数据按折数分层排列
    X = [X1, X2]  # 融合一维数据和多维数据
    X, y = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)

    train_point = int(len(X) * 0.2) if type(X) != list else int(len(X[0]) * 0.2)

    # 融合分类器训练
    model = FS_Clf(learning_rate=0.001, epochs=100, batch_size=100, random_state=seed, feat_len_list=feat_len_list,
                   re_size=re_size)
    model.param_search = False
    model.only_save_last_epoch = True
    model.metrics = accuracy_score
    model.metrics_list = [f1_macro_score, f1_weighted_score]
    model.fit([x[train_point:] for x in X], y[train_point:], [x[:train_point] for x in X], y[:train_point])
