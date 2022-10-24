# -*- coding: utf-8 -*-
# @Time    : 2021/11/15 下午8:47
# @Author  : CSH_Yatpit

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from urllib.parse import urlparse

import pkg_resources
import tldextract
from bs4 import BeautifulSoup

from skimage import io, transform
from symspellpy import SymSpell

from optUtils import make_dirs


def scheme(url):
    # 解析url
    res = urlparse(url)
    url_scheme = res.scheme
    # 返回 http or https
    return url_scheme


# 获取钓鱼url的域名
def domain(url):
    # 解析url
    res = urlparse(url)
    url_domain = res.netloc
    # print(res)
    # print(url_domain.count('-'))
    return url_domain


# 获取钓鱼url的二级域名
def second_domain(url):
    # 解析url，提取子域名
    subdomain = tldextract.extract(url).domain
    # 解析url，提取顶级域名
    suffix = tldextract.extract(url).suffix
    # print(suffix)
    # 把子域名和顶级域名拼起来
    second_domain = subdomain + '.' + suffix
    # print(second_domain)
    return second_domain


# 获取url的顶级域名
def top_domain(url):
    top_domain = tldextract.extract(url).suffix
    return top_domain


# 获取url的域名中'-'的个数
def domain_dash_count(url):
    dash_count = 0
    res = urlparse(url)
    url_domain = res.netloc
    dash_count = url_domain.count('-')
    # url_domain.count('_')
    return dash_count


def domain_symbol_count(url):
    dash_count = 0
    res = urlparse(url)
    url_domain = res.netloc
    print(url_domain)
    dash_count = url_domain.count('-')
    # url_domain.count('_')
    return dash_count


# 获取url的域名数字的个数
def domain_num_count(url):
    num_count = 0
    res = urlparse(url)
    url_domain = res.netloc
    for i in url_domain:
        if i.isdigit():
            num_count = num_count + 1
    return num_count


# 获取url中特殊符号的个数
def special_symbol_count(url):
    special_sym_count = 0
    # print(url_domain)
    token_list = ['-', '.', '/', ':']
    for i in url:
        # 计算url中的特殊符号，不包括'-', '.', '/', ':'
        if (i.isalnum() == False) & (i not in token_list):
            # print(url)
            special_sym_count = special_sym_count + 1
            # print(i)
    return special_sym_count


# 获取url中域名后面的长度
def domain_len(url):
    res = urlparse(url)
    url_domain = res.netloc
    return len(url_domain)


def behind_domain_len(url):
    url_str = ''.join(url.split('/')[3:])
    return len(url_str)


# 获取url中域名后面'/'的个数
def slash_count(url):
    slash_count = url.count('/') - 2
    return slash_count


# 获取url中个数最多的字符
def top_char(url):
    char_count_dict = {}
    for c in url:
        char_count_dict[c] = char_count_dict.get(c, 0) + 1
    return sorted(char_count_dict.items(), key=lambda x: x[-1], reverse=True)[0][0]


# 获取url中个数最多的特殊字符
def top_special(url):
    special_dict = {}
    token_list = ['-', '.', '/', ':']
    for c in url:
        # 计算url中的特殊符号，不包括'-', '.', '/', ':'
        if c.isalnum() is False and c not in token_list:
            special_dict[c] = special_dict.get(c, 0) + 1
    special_list = sorted(special_dict.items(), key=lambda x: x[-1], reverse=True)
    return special_list[0][0] if special_list else np.nan


# 传入品牌名、品牌数量表，把样本数量小于20的品牌改为‘Other’
def change_target(target, info):
    if info[target] < 10:
        target = 'Other'
    return target


# 传入品牌名、品牌数量表，把样本数量小于20的品牌删除
def drop_target(target, info):
    if info[target] < 5:
        target = 0  # 把target填为0,下一步判断删除
    return target


# 传入域名创建时间和过期时间，计算域名有效年限
def valid_days(creation, expiration):
    # 域名创建时间和过期时间为空，不执行
    if (type(creation) != float) & (type(expiration) != float):
        # 获取日期
        creation = creation.split(' ')[0]
        expiration = expiration.split(' ')[0]
        # 将日期转为时间格式
        creation = datetime.strptime(creation, '%Y/%m/%d')
        expiration = datetime.strptime(expiration, '%Y/%m/%d')
        # 计算域名有效年限
        # print(creation)
        # print(expiration)
        valid_day = int((expiration - creation).days)
        # print(valid_day)
        return valid_day


# 当num为1，提取A.B.C；当num为2,提取A.B
def split_A(A, num):
    if type(A) != float:
        # 反向切取
        A_res = A.rsplit('.', num)[0]
        return A_res


# 把CNAME统一转成小写
def cname(cname):
    if type(cname) != float:
        cname = cname.lower()
    return cname


# 把icon转成字符
def icon_str(icon_path):
    if type(icon_path) != float:
        icon_path = './%s/%s' % (data_dir, icon_path)
        # print(icon_path)
        if os.path.exists(icon_path):
            try:
                img_arr = io.imread(icon_path)
                # print(img.shape)
                # img_arr = img.reshape((16, 16, 4))
                # print(img_arr)
                # img = Image.open(icon_path)
            except:
                return
            # img_arr = np.array(img)
            # 把数组转成字节再转成十六进制
            img_str = img_arr.tobytes().hex()
            return img_str


# def cut_words_path(url):
#     print('ooo')
#     res = urlparse(url)
#     pattern = '[/.?@#_:]'
#     text = res.netloc + res.path
#     text = re.sub(pattern, ' ', text)
#     # print(text)
#     sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
#     dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
#     sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
#     result = sym_spell.word_segmentation(text, max_segmentation_word_length=14)
#     print(result.segmented_string)
#     return result.corrected_string


def strip_space(url_words):
    # print(url_words)
    # url_words.strip()
    print(url_words.strip())
    print('\n')
    return url_words


def top_url_word(url_words):
    url_words_dict = {}
    url_words_list = url_words.split()
    print(url_words_list)
    for word in url_words_list:
        url_words_dict[word] = url_words_dict.get(word, 0) + 1
    url_words_dict = sorted(url_words_dict.items(), key=lambda x: x[-1], reverse=True)
    sort_list = sorted(map(lambda x: x[0], url_words_dict[:3]), key=lambda x: x)
    # url_words_dict = sorted(url_words_list, key=lambda x: x)
    # top3_words = ' '.join(url_words_dict[:3])
    top3_words = ' '.join(sort_list[:3])
    print(top3_words)
    return top3_words


def url_word_top3(url_words, url_word_dict):
    if url_words and url_words is not np.nan:
        url_words = sorted(set(url_words), key=lambda x: [url_word_dict[x], x], reverse=True)
        return ",".join(url_words[:3])


# 清洗html中的文本
def all_words(html_path):
    if type(html_path) != float:
        html_path = './%s/%s' % (data_dir, html_path)
        with open(html_path, 'r', encoding='utf-8-sig', errors='ignore') as strf:
            str_html = strf.read()
        # 获取html中的文本，过滤转义符
        soup = BeautifulSoup(str_html).get_text()
        soup = soup.replace('\t', ' ')
        soup = soup.replace('\n', ' ')
        # 正则过滤中日韩文和特殊字符
        pattern = '[\u4e00-\u9fa5\u3040-\u31FF\uAC00-\uD7AF\t,0-9\'!;%?`*=#$.|✓\\<>:—•+。、"/○○●@✔©_&＋％＆▲~»…｜〜！\[\]' \
                  '【】『』()（）「」《》“”{}，–-]'
        text = re.sub(pattern, '', soup).strip().lower()
        if text != '':
            # text = cut_words(text)
            str_list = text.split()
            return str_list
        else:
            return


# 计算单词idf
def get_idf_dict(html_words_lists):
    idf_dict = dict()
    # 计算每个样本中每个单词在所有样本中的出现次数
    for html_words_list in html_words_lists:
        html_word_unique = set(html_words_list)
        # 遍历每个样本中的单词
        for word in html_word_unique:
            idf_dict[word] = idf_dict.get(word, 0) + 1
    # 计算每个单词的idf
    idf_dict = {word: np.log(len(html_words_lists) / value) for word, value in idf_dict.items()}
    return idf_dict


# 计算单词tf-idf
def cal_tfidf(dataset, html_words_lists, idf_dict):
    for ix in html_words_lists.index:
        tf_dict = dict()
        # 计算每个单词在该样本中的出现次数
        for word in html_words_lists[ix]:
            tf_dict[word] = tf_dict.get(word, 0) + 1
        tfidf_dict = {word: value / len(html_words_lists[ix]) * idf_dict[word] for word, value in tf_dict.items()}
        # 取前三个tf-idf值最大的单词
        sort_dict = sorted(tfidf_dict.items(), key=lambda x: x[-1], reverse=True)
        sort_list = sorted(map(lambda x: x[0], sort_dict[:3]), key=lambda x: x)
        dataset.loc[ix, 'tfidf_top3'] = str(sort_list)
    return dataset


def cal_tf(dataset, html_words_lists):
    for ix in html_words_lists.index:
        tf_dict = dict()
        for word in html_words_lists[ix]:
            tf_dict[word] = tf_dict.get(word, 0) + 1
        sort_dict = sorted(tf_dict.items(), key=lambda x: x[-1], reverse=True)
        sort_list = sorted(map(lambda x: x[0], sort_dict[:3]), key=lambda x: x)
        dataset.loc[ix, 'tf'] = str(sort_list)
    return dataset


# 提取钓鱼url中包含的敏感词
def sens_words_url(url):
    sens_words_lists = ['token', 'confirm', 'security', 'log', 'sign', 'login', 'signin', 'bank', 'account',
                        'update', 'ebay', 'secure']
    url_sens_words = []
    for i in range(len(sens_words_lists)):
        if sens_words_lists[i] in url:
            url_sens_words.append(sens_words_lists[i])
    if str(url_sens_words) == '[]':
        return None
    else:
        return ','.join(url_sens_words)


# 提取钓鱼url中domain包含的敏感词
def sens_words_domain(url):
    res = urlparse(url)
    sens_words_lists = ['token', 'confirm', 'security', 'log', 'sign', 'login', 'signin', 'bank', 'account',
                        'update', 'ebay', 'secure']
    sens_words = []
    for i in range(len(sens_words_lists)):
        if sens_words_lists[i] in res.hostname:
            sens_words.append(sens_words_lists[i])
    # 过滤空列表
    if str(sens_words) == '[]':
        return None
    else:
        return ','.join(sens_words)


# 提取钓鱼url中path包含的敏感词
def sens_words_path(url):
    res = urlparse(url)
    sens_words_lists = ['token', 'confirm', 'security', 'log', 'sign', 'login', 'signin', 'bank', 'account',
                        'update', 'ebay', 'secure']
    sens_words = []
    for i in range(len(sens_words_lists)):
        if sens_words_lists[i] in res.path:
            sens_words.append(sens_words_lists[i])
    # 过滤空列表
    if str(sens_words) == '[]':
        return None
    else:
        return ','.join(sens_words)


# 提取html文本中包含的品牌词
def brand_words_html(brand_list, html_list):
    brand_dict = {brand: re.sub("[,.()]", " ", brand.lower()).split() for brand in brand_list}
    for brand in brand_dict:
        brand_word_list = brand_dict[brand]
        if len(brand_word_list) >= 3:
            brand_dict[brand] += ["".join([brand_word[0] for brand_word in brand_word_list])]
    if html_list is None:
        return None
    else:
        brand_count_dict = {}
        word_count_dict = {}
        for word in html_list:
            word_count_dict[word] = word_count_dict.get(word, 0) + 1
            for brand in brand_dict:
                for brand_word in brand_dict[brand]:
                    if brand_word == word:
                        brand_count_dict[brand] = brand_count_dict.get(brand, 0) + 1
        brand_count_list = sorted(brand_count_dict.items(), key=lambda x: x[-1], reverse=True)
        if brand_count_list:
            return brand_count_list[0][0]
        else:
            return sorted(word_count_dict.items(), key=lambda x: x[-1], reverse=True)[0][0]
    # # 过滤空列表
    # if str(brand_words) == '[]':
    #     return None
    # else:
    #     # return ','.join(brand_words)
    #     return brand_words[0]


# 提取html文本中包含的敏感词
def sens_words_html(html_list):
    if html_list is None:
        return None
    else:
        sens_words_lists = ['token', 'confirm', 'security', 'log', 'sign', 'login', 'signin', 'bank', 'account',
                            'update', 'ebay', 'secure']
        html_sens_words = []
        for i in range(len(sens_words_lists)):
            if sens_words_lists[i] in set(html_list):
                html_sens_words.append(sens_words_lists[i])
        if str(html_sens_words) == '[]':
            return None
        else:
            return ','.join(html_sens_words)


# 计算dig-A信息中IP的个数
def A_IP_num(digA, phish_id):
    # print(phish_id)
    ip_count = digA.value_counts('phish_id')
    # print(ip_count)
    if phish_id in ip_count.index.tolist():
        ip_num = int(ip_count[ip_count.index == phish_id].values)
        # print(type(ip_num),'\n')
        return ip_num


# 计算域名级数
def domain_level(url):
    res = urlparse(url)
    subdomain = res.netloc
    # print(subdomain.count('.'))
    return subdomain.count('.')


# 存ocr结果的txt路径
def ocr_result(img_path):
    if type(img_path) == float:
        return
    else:
        dir_path = './%s/picture_ocr' % data_dir
        file_name = img_path.split('.')[0].split('/')[-1]
        file_path = dir_path + '/' + file_name + '.txt'
        if os.path.exists(file_path):
            # print(file_path)
            return file_path
        else:
            return


# 提取ocr结果中包含的品牌词
def brand_words_ocr(txt_path, brand_list):
    brand_dict = {brand: re.sub("[,.()]", " ", brand.lower()).split() for brand in brand_list}
    for brand in brand_dict:
        brand_word_list = brand_dict[brand]
        if len(brand_word_list) >= 3:
            brand_dict[brand] += ["".join([brand_word[0] for brand_word in brand_word_list])]
    # ocr_brand_words = []
    if txt_path is None:
        # print(txt_path)
        return None
    else:
        # print(txt_path)
        with open(txt_path, 'r') as f:
            word_list = f.readlines()
        brand_count_dict = {}
        word_count_dict = {}
        for word in word_list:
            word = word.strip()
            word_count_dict[word] = word_count_dict.get(word, 0) + 1
            for brand in brand_dict:
                for brand_word in brand_dict[brand]:
                    if brand_word == word:
                        brand_count_dict[brand] = brand_count_dict.get(brand, 0) + 1
        brand_count_list = sorted(brand_count_dict.items(), key=lambda x: x[-1], reverse=True)
        if brand_count_list:
            return brand_count_list[0][0]
        else:
            return sorted(word_count_dict.items(), key=lambda x: x[-1], reverse=True)[0][0]


def sens_words_ocr(txt_path):
    sens_words_lists = ['token', 'confirm', 'security', 'log', 'sign', 'login', 'signin', 'bank', 'account',
                        'update', 'ebay', 'secure']
    if txt_path is None:
        # print(txt_path)
        return None
    else:
        with open(txt_path, 'r') as f:
            word_list = f.readlines()
            for word in word_list:
                word = word.strip()
                if word in sens_words_lists:
                    return word
        return None


def cut_words_domain(url):
    # print(url)
    res = urlparse(url)
    pattern = '[/.?@#_:]'
    text = res.hostname
    text = re.sub(pattern, ' ', text)
    print(text, '111')
    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    result = sym_spell.word_segmentation(text, max_segmentation_word_length=14)
    print(result.segmented_string)
    return result.corrected_string


def cut_words_path(url):
    # print(url)
    res = urlparse(url)
    pattern = '[/.?@#_:]'
    text = res.path
    text = re.sub(pattern, ' ', text)
    print(text, '111')
    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    result = sym_spell.word_segmentation(text, max_segmentation_word_length=14)
    print(result.segmented_string)
    return result.corrected_string


def url_digit_count(url):
    re = urlparse(url)
    url_str = re.netloc + re.path
    count = 0
    for i in url_str:
        if i.isdigit():
            count = count + 1
    print(count)
    return count


def tag_count(html_path):
    if type(html_path) != float:
        html_path = './%s/%s' % (data_dir, html_path)
        tag_counts = 0
        with open(html_path, encoding='utf-8', errors='ignore') as f:
            html1 = f.read()
        re_script = '<([a-zA-Z]{1,})'
        pattern = re.findall(re_script, html1)
        # print(pattern)
        pattern_dict = {}
        for tag in pattern:
            pattern_dict[tag] = pattern_dict.get(tag, 0) + 1
        # print(pattern_dict)
        for k, v in pattern_dict.items():
            if k in ['link', 'script', 'form', 'img', 'a', 'image']:
                tag_counts = tag_counts + pattern_dict[k]
        return tag_counts


def html_text_symbol(html_path):
    if type(html_path) != float:
        html_path = './dataset/' + html_path
        with open(html_path, 'r', encoding='utf-8-sig', errors='ignore') as strf:
            str_html = strf.read()
        # 获取html中的文本，过滤转义符
        soup = BeautifulSoup(str_html).get_text()
        soup = soup.replace('\t', ' ')
        soup = soup.replace('\n', ' ')
        # pattern = '[!;%?`*=#$.|✓\\<>:—•+。、"/○○●@✔©_&＋％＆▲~»…｜〜！\[\]【】『』()（）「」《》“”{}，–-]'
        pattern = '[\uFF00-\uFFEF]'
        text = re.findall(pattern, soup)
        return len(text)
        # if text != '':
        #     # text = cut_words(text)
        #     str_list = text.split()
        #     print(str_list)
        #     # print(len(str_list))
        #     return len(str_list)
        # else:
        #     return


def create_train_dataset(dataset):
    # 获取品牌名称表和品牌数量表
    brand_num = dataset['target'].unique()
    print('原始品牌数', len(brand_num))
    info = dataset['target'].value_counts()
    print('原始样本数量：', len(dataset['phish_id'].value_counts()))

    # 存钓鱼url的协议
    print('存钓鱼url的协议')
    dataset['scheme'] = dataset.apply(lambda x: scheme(x['url']), axis=1)
    print('OK...')

    # 存钓鱼url的域名
    print('存钓鱼url的域名')
    dataset['domain'] = dataset.apply(lambda x: domain(x['url']), axis=1)
    print('OK...')

    print("存钓鱼url的顶级级域名")
    dataset['top_domain'] = dataset.apply(lambda x: top_domain(x['url']), axis=1)
    print('OK...')

    # 存钓鱼url的二级域名
    print("存钓鱼url的二级域名")
    dataset['second_domain'] = dataset.apply(lambda x: second_domain(x['url']), axis=1)
    print('OK...')

    # 存域名级数
    print("存域名级数")
    dataset['domain_level'] = dataset.apply(lambda x: domain_level(x['url']), axis=1)
    print('OK...')

    # 获取url的域名中'-'个数
    dataset['domain_dash_count'] = dataset.apply(lambda x: domain_dash_count(x['url']), axis=1)

    # 获取url的域名数字的个数
    dataset['domain_num_count'] = dataset.apply(lambda x: domain_num_count(x['url']), axis=1)

    # 获取url中域名后面'/'的个数
    dataset['slash_count'] = dataset.apply(lambda x: (slash_count(x['url'])), axis=1)

    # 获取url中特殊符号的个数
    dataset['special_symbol_count'] = dataset.apply(lambda x: special_symbol_count(x['url']), axis=1)

    # 获取url中域名后面的字符串长度
    dataset['domain_len'] = dataset.apply(lambda x: (domain_len(x['url'])), axis=1)

    dataset['behind_domain_len'] = dataset.apply(lambda x: (behind_domain_len(x['url'])), axis=1)

    # 获取url中个数最多的字符
    dataset['top_char'] = dataset.apply(lambda x: top_char(x['url']), axis=1)
    # 获取url中个数最多的特殊字符
    dataset['top_special'] = dataset.apply(lambda x: top_special(x['url']), axis=1)

    # 存域名的有效天数
    print("存域名的有效天数")
    dataset['valid_days'] = dataset.apply(lambda x: valid_days(x['creation_date'], x['expiration_date']), axis=1)
    print('OK...')

    # 存dig-A信息的IP
    print("存dig-A信息的IP")
    # A_1 = A.B.C
    dataset['A_1'] = dataset.apply(lambda x: split_A(x['A'], 1), axis=1)
    # A_2 = A.B
    dataset['A_2'] = dataset.apply(lambda x: split_A(x['A'], 2), axis=1)
    print('OK...')

    # 存dig-A信息的IP个数
    print("存dig-A信息的IP个数")
    dataset['A_IP_num'] = dataset.apply(lambda x: A_IP_num(digA, x['phish_id']), axis=1)
    print('OK...')

    # 存dig-A信息的CNAME，CNAME统一小写
    print("存dig-A信息的CNAME")
    dataset['CNAME'] = dataset.apply(lambda x: cname(x['CNAME']), axis=1)
    print('OK...')

    # 删除数量少的target样本
    dataset['target'] = dataset.apply(lambda x: drop_target(x['target'], info), axis=1)
    dataset = dataset[~dataset['target'].isin([0])].reset_index(drop=True)
    info1 = dataset['target'].value_counts()
    info2 = dataset['phish_id'].value_counts()
    print('最终品牌数：', len(info1))
    print('最终样本数量：', len(info2))

    # 存由icon转化的十六进制字符串
    print("存由icon转化的十六进制字符串")
    dataset['icon_str'] = dataset.apply(lambda x: icon_str(x['icon']), axis=1)
    print('OK...')

    # 从csv中获取url单词表，拼接到dataset
    print("存url切词")
    url_words = pd.read_csv('url_words_result.csv').loc[:, ['phish_id', 'url_words']]
    dataset = dataset.merge(url_words, on='phish_id', how='left')

    def temp(x):
        if x is not np.nan:
            return x.split()

    dataset['url_words'] = dataset['url_words'].apply(lambda x: temp(x))
    url_word_dict = {}
    for word_list in dataset['url_words']:
        if word_list:
            for word in word_list:
                url_word_dict[word] = url_word_dict.get(word, 0) + 1

    # dataset['url_words'] = dataset.apply(lambda x: strip_space(x['url_words']), axis=1)
    # 存url中出现频率最高的单词
    # dataset['top_url_word'] = dataset.apply(lambda x: top_url_word(x['url_words']), axis=1)

    dataset['url_word_top3'] = dataset.apply(lambda x: url_word_top3(x['url_words'], url_word_dict), axis=1)
    print('OK...')

    # 存钓鱼url的中包含的敏感词
    print("存钓鱼url中包含的敏感词")
    dataset['sens_words_url'] = dataset.apply(lambda x: (sens_words_url(x['url'])), axis=1)
    print('OK...')

    # 存html文本中包含的单词列表
    print("存html文本中包含的单词列表")
    dataset['all_words'] = dataset.apply(lambda x: all_words(x['html']), axis=1)
    print('OK...')

    # 存钓鱼html文本中包含的敏感词
    print("存钓鱼html文本中包含的敏感词")
    dataset['sens_words_html'] = dataset.apply(lambda x: sens_words_html(x['all_words']), axis=1)
    print('OK...')

    # 存ocr结果的路径
    print('存ocr结果的路径')
    dataset['ocr_result'] = dataset.apply(lambda x: ocr_result(x['picture']), axis=1)
    print('OK...')

    # 存ocr结果中包含的品牌名
    print('存ocr结果中包含的品牌名 ')
    dataset['sens_words_ocr'] = dataset.apply(lambda x: sens_words_ocr(x['ocr_result']), axis=1)
    print('OK...')

    # 存钓鱼html文本中包含的品牌词
    print("存钓鱼html文本中包含的品牌词")
    dataset['brand_words_html'] = dataset.apply(lambda x: brand_words_html(brand_num, x['all_words']), axis=1)
    print('OK...')

    dataset['brand_words_ocr'] = dataset.apply(lambda x: brand_words_ocr(x['ocr_result'], brand_num), axis=1)

    # 取非空白的单词列表
    print('取非空白的单词列表')
    html_words_lists = dataset.loc[
        (~dataset['all_words'].isna()) & (dataset['all_words'].astype('str') != '[]')].all_words
    print('OK...')

    # 计算html文本中单词的tf-idf
    print("计算html文本中单词的tf-idf")
    idf_dict = get_idf_dict(html_words_lists)
    dataset = cal_tfidf(dataset, html_words_lists, idf_dict)
    print('OK...')

    print('html中特定标签的个数')
    dataset['tag_count'] = dataset.apply(lambda x: (tag_count(x['html'])), axis=1)

    dataset['html_text_symbol'] = dataset.apply(lambda x: (html_text_symbol(x['html'])), axis=1)
    print('OK...')

    # print("存钓鱼url的domain中包含的敏感词")
    # dataset['sens_words_domain'] = dataset.apply(lambda x: (sens_words_domain(x['url'])), axis=1)
    # print('OK...')

    # # 存钓鱼url的path中包含的敏感词
    # print("存钓鱼url的path中包含的敏感词")
    # dataset['sens_words_path'] = dataset.apply(lambda x: sens_words_path(x['url']), axis=1)
    # print('OK...')

    # dataset['url_domain_words'] = dataset.apply(lambda x: (cut_words_domain(x['url'])), axis=1)
    # dataset['url_path_words'] = dataset.apply(lambda x: (cut_words_path(x['url'])), axis=1)
    # dataset['avg_url_words'] = dataset.apply(lambda x: (avg_url_words(x['url_words'])), axis=1)
    # dataset['url_digit_count'] = dataset.apply(lambda x: (url_digit_count(x['url'])), axis=1)
    # 存html中特定标签的个数

    return dataset


# 数据集文件夹修改
data_dir = "dataset"

# 读取csv
dataset = pd.read_csv(data_dir + '/phishing-target.csv', index_col=None, encoding='utf-8')
digA = pd.read_csv(data_dir + '/phishing-A.csv', index_col=None)
digCname = pd.read_csv(data_dir + '/phishing-CNAME.csv', index_col=None)

print(len(dataset['target'].unique()))
# print(len(digCname['phish_id'].unique()))

# 拼接三个表的pd数据，去重
dataset = dataset.merge(digA, on='phish_id', how='left')
dataset = dataset.merge(digCname, on='phish_id', how='left')
dataset = dataset.drop_duplicates(subset='phish_id')

dataset = create_train_dataset(dataset)


def picture_data(dataset, pic_type, re_size=64):
    picture_path_list = list(dataset[pic_type])
    resize_picture_dir = "./{}/{}".format(data_dir, pic_type + str(re_size))
    save_picture = False
    if not os.path.exists(resize_picture_dir):
        make_dirs(resize_picture_dir)
        save_picture = True

    img_list = []
    for picture_path in picture_path_list:
        if picture_path is not np.nan:
            picture_path = "./{}/{}".format(data_dir, picture_path)
            if not save_picture:
                picture_path = resize_picture_dir + '/' + picture_path.split('/')[-1]
            if os.path.exists(picture_path):
                try:
                    img = io.imread(picture_path)
                    img = img / 255.
                    if save_picture:
                        img = transform.resize(img, (re_size, re_size, 3))
                        io.imsave(resize_picture_dir + '/' + picture_path.split('/')[-1], img)
                except:
                    print("except:", picture_path)
                    img = np.zeros((re_size, re_size, 3)).astype('float') + 0.5
            else:
                print("dont exist:", picture_path)
                img = np.zeros((re_size, re_size, 3)).astype('float') + 0.5
        else:
            print("dont have path:", picture_path)
            img = np.zeros((re_size, re_size, 3)).astype('float') + 0.5
        img_list.append(img)
    return np.array(img_list)
