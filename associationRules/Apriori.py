# -*- encoding: utf-8 -*-
# @Version : 1.0  
# @Time    : 2019/5/7 16:55
# @Author  :  fireWang
# @note    :  使用关联规则发现毒蘑菇数据集中 毒蘑菇的特征

import os
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def get_data():
    # 读取数据
    data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'agaricus-lepiota.data'), header=None)
    # 筛选出毒蘑菇
    data = data.loc[data.iloc[:, 0] == 'p', 1:]
    # 重置下行索引
    data.reset_index(drop=True, inplace=True)
    # 将数据转化为 热编码
    data = pd.get_dummies(data)
    return data


if __name__ == '__main__':
    # 获取数据
    data = get_data()
    # 打印下数据的维度
    print(data.shape)
    # 看下数据
    print(data.head())
    # 发现频繁项集
    frequent_sets = apriori(data, min_support=0.7, use_colnames=True, max_len=2)
    # 基于频繁项集 生成关联规则
    rules = association_rules(frequent_sets, min_threshold=1)
    # 输出到 excel
    rules.to_excel('./data/rules.xlsx',index=False)
