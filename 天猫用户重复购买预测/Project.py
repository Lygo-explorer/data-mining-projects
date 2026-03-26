import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import VarianceThreshold  # 根据数据方差特征选择
from sklearn.feature_selection import SelectKBest  # 单变量特征选择
from sklearn.feature_selection import mutual_info_classif  # 分类模型的特征评价函数
from sklearn.feature_selection import SelectFromModel  # 用模型选择特征
from sklearn.feature_selection import RFECV  # 递归功能消除
from sklearn.linear_model import LogisticRegression  # 回归分类模型
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import ExtraTreesClassifier  # 树模型
from sklearn.preprocessing import StandardScaler  # 归一化
from sklearn.model_selection import train_test_split  # 简单验证
from sklearn.model_selection import KFold  # k折验证
from sklearn.model_selection import ShuffleSplit  # 随机取样
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.metrics import classification_report  # 分类效果报告
from sklearn.metrics import confusion_matrix  #混淆矩阵
import warnings
from collections import Counter
warnings.filterwarnings("ignore")


# 数据探索---------------------------------------------------------------------------------------------------------------
# 读取数据
# test_data = pd.read_csv('data_format1/data_format1/test_format1.csv')
# train_data = pd.read_csv('data_format1/data_format1/train_format1.csv')
# user_info = pd.read_csv('data_format1/data_format1/user_info_format1.csv')
# user_log = pd.read_csv('data_format1/data_format1/user_log_format1.csv')


# 查看数据情况并处理缺失值
# 查看数据情况
# print('head--------------')
# print('test_data')
# print(test_data.head(5))
# print('train_data')
# print(train_data.head(5))
# print('user_info')
# print(user_info.head(5))
# print('user_log')
# print(user_log.head(5))
# print('info--------------')
# print('test_data')
# print(test_data.info())
# print('train_data')
# print(train_data.info())
# print('user_info')
# print(user_info.info())
# print('user_log')
# print(user_log.info())
# 通过info发现user_info.gender和user_info.age_range有数据缺失
# 查看数据分布情况
# print('describe--------------')
# print('test_data')
# print(test_data.describe())
# print('train_data')
# print(train_data.describe())
# print('user_info')
# print(user_info.describe())
# print('user_log')
# print(user_log.describe())
# 查看数据正负样本分布情况
# label_gp = train_data.groupby('label')['user_id'].count()
# print('正负样本数量:\n', label_gp)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# train_data.label.value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, explode=[0, 0.1])
# plt.subplot(1, 2, 2)
# sns.countplot(x='label', data=train_data)
# plt.show()

# 发掘缺失值
# print('user_info.gender缺失率：', (user_info.shape[0] - user_info['gender'].count())/user_info.shape[0])
# print('user_info.age_range缺失率', (user_info.shape[0] - user_info['age_range'].count())/user_info.shape[0])
# print(user_info[user_info['age_range'].isna() | (user_info['age_range'] == 0) | user_info['gender'].isna() |
#           (user_info['gender'] == 2)].shape[0])
# print(user_log.isna().sum())
# user_info.gender缺失率： 0.01517316170403376
# user_info.age_range缺失率 0.005226677982884221
# age_range为0或缺失或者gender为2或缺失的样本数为106330
# user_log.brand_id缺失样本有91015


# 变量分析并处理异常值
# 变量与目标关系分析
# 以用户性别为例
# train_data_user_info = train_data.merge(user_info, on=['user_id'])
# plt.figure(figsize=(8, 8))
# plt.title('Gender VS Label')
# sns.countplot(x='gender', hue='label', data=train_data_user_info)
# plt.show()
# repeat_buy = [rate for rate in train_data_user_info.groupby('gender')['label'].mean()]
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# sns.distplot(repeat_buy, fit=stats.norm)
# plt.subplot(1, 2, 2)
# stats.probplot(repeat_buy, plot=plt)
# plt.show()


# 特征工程---------------------------------------------------------------------------------------------------------------
# 合并数据
# all_data = train_data._append(test_data)
# all_data = all_data.merge(user_info, on=['user_id'], how='left')
# del train_data, test_data, user_info
# 数据处理
# user_log = user_log.sort_values(['user_id', 'time_stamp'])  # 将user_log数据按照固定属性排序
# list_join_func = lambda x: " ".join([str(i) for i in x])  # 将数据变为字符串类型
# agg_dict = {
#     'item_id': list_join_func,
#     'cat_id': list_join_func,
#     'seller_id': list_join_func,
#     'brand_id': list_join_func,
#     'time_stamp': list_join_func,
#     'action_type': list_join_func
# }
# rename_dict = {
#     'item_id': 'item_path',
#     'cat_id': 'cat_path',
#     'seller_id': 'seller_path',
#     'brand_id': "brand_path",
#     'time_stamp': 'time_stamp_path',
#     'action_type': 'action_type_path'
# }  # 将属性改名
# 将user_log合并入all_data
# def merge_list(df_ID, join_columns, df_data, agg_dict, rename_dict):
#     df_data = df_data.groupby(join_columns).agg(agg_dict).reset_index().rename(columns=rename_dict)
#     df_ID = df_ID.merge(df_data, on=join_columns, how='left')
#     return df_ID
# all_data = merge_list(all_data, 'user_id', user_log, agg_dict, rename_dict)


# 特征构造
# 人为构造特征
# 定义统计函数
# def cnt_(x):  # 统计数据总数
#     try:
#         return len(x.split(' '))
#     except:
#         return -1
# def nunique_(x):  # 统计数据唯一值总数
#     try:
#         return len(set(x.split(' ')))
#     except:
#         return -1
# def max_(x):  # 统计数据最大值
#     try:
#         return np.max([float(i) for i in x.split(' ')])
#     except:
#         return -1
# def min_(x):  # 统计数据最小值
#     try:
#         return np.min([float(i) for i in x.split(' ')])
#     except:
#         return -1
# def std_(x):  # 数据标准差
#     try:
#         return np.std([float(i) for i in x.split(' ')])
#     except:
#         return -1
# def most_n(x, n):  # 统计数据中topN数据
#     try:
#         return Counter(x.split(' ')).most_common(n)[n-1][0]
#     except:
#         return -1
# def most_n_cnt(x, n):  # 统计数据中topN数据总数
#     try:
#         return Counter(x.split(' ')).most_common(n)[n-1][1]
#     except:
#         return -1
# 构造特征
# def user_cnt(df_data, single_col, name):
#     df_data[name] = df_data[single_col].apply(nunique_)
#     return df_data
# def user_nunique(df_data, single_col, name):
#     df_data[name] = df_data[single_col].apply(nunique_)
#     return df_data
# def user_max(df_data, single_col, name):
#     df_data[name] = df_data[single_col].apply(max_)
#     return df_data
# def user_min(df_data, single_col, name):
#     df_data[name] = df_data[single_col].apply(min_)
#     return df_data
# def user_std(df_data, single_col, name):
#     df_data[name] = df_data[single_col].apply(std_)
#     return df_data
# def user_most_n(df_data, single_col, name, n=1):
#     func = lambda x: most_n(x, n)
#     df_data[name] = df_data[single_col].apply(func)
#     return df_data
# def user_most_n_cnt(df_data, single_col, name, n=1):
#     func = lambda x: most_n_cnt(x, n)
#     df_data[name] = df_data[single_col].apply(func)
#     return df_data
# all_data_test = all_data.loc[0:1000, :]
# all_data_test = user_cnt(all_data_test, 'seller_path', 'user_cnt')
# all_data_test = user_nunique(all_data_test, 'seller_path', 'user_cnt')
# all_data_test = user_nunique(all_data_test, 'cat_path', 'cat_nunique')
# all_data_test = user_nunique(all_data_test, 'brand_path', 'brand_nunique')
# all_data_test = user_nunique(all_data_test, 'item_path', 'item_nunique')
# all_data_test = user_nunique(all_data_test, 'time_stamp_path', 'time_stamp_nunique')
# all_data_test = user_nunique(all_data_test, 'action_type_path', 'action_type_nunique')
# all_data_test = user_max(all_data_test, 'action_type_path', 'time_stamp_max')
# all_data_test = user_min(all_data_test, 'action_type_path', 'time_stamp_min')
# all_data_test = user_std(all_data_test, 'action_type_path', 'time_stamp_std')
# all_data_test['time_stamp_range'] = all_data_test['time_stamp_max'] - all_data_test['time_stamp_min']
# all_data_test = user_most_n(all_data_test, 'seller_path', 'seller_most_1', n=1)
# all_data_test = user_most_n(all_data_test, 'cat_path', 'cat_most_1', n=1)
# all_data_test = user_most_n(all_data_test, 'brand_path', 'brand_most_1', n=1)
# all_data_test = user_most_n(all_data_test, 'action_type_path', 'action_type_1', n=1)
# all_data_test = user_most_n_cnt(all_data_test, 'seller_path', 'seller_most_1_cnt', n=1)
# all_data_test = user_most_n_cnt(all_data_test, 'cat_path', 'cat_most_1_cnt', n=1)
# all_data_test = user_most_n_cnt(all_data_test, 'brand_path', 'brand_most_1_cnt', n=1)
# all_data_test = user_most_n_cnt(all_data_test, 'action_type_path', 'action_type_1_cnt', n=1)
# print(all_data_test.columns)
# all_data_test.to_csv('data_test.txt')


# 特征选择
# 读取数据
data = pd.read_csv('data_test.txt')
data.drop(['prob', data.columns[0]], axis=1, inplace=True)
data.dropna(inplace=True)
target = data['label'].values
data.drop(['label', 'item_path', 'cat_path', 'seller_path', 'brand_path', 'time_stamp_path', 'action_type_path'],
          inplace=True, axis=1)
train = data.values

# 去除方差较小的特征
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# train_sel = sel.fit_transform(train)

# 单变量特征选择
# sel = SelectKBest(mutual_info_classif, k=17)
# train_sel = sel.fit_transform(train_sel, target)

# 递归功能消除
# clf = RandomForestClassifier(n_estimators=10, max_depth=2, n_jobs=-1)
# sel = RFECV(clf, step=1, cv=2)
# train_sel = sel.fit_transform(train_sel, target)

# 用模型选择特征
# clf = ExtraTreesClassifier(n_estimators=5)
# clf = clf.fit(train, target)
# model = SelectFromModel(clf, prefit=True)
# train_sel = model.transform(train)


# 模型训练---------------------------------------------------------------------------------------------------------------
# kf = KFold(n_splits=5)
# for k, (train_index, test_index) in enumerate(kf.split(train)):
#     train_data, test_data, train_target, test_target = train[train_index], train[test_index], target[train_index], \
#         target[test_index]
#     clf = LogisticRegression()
#     clf.fit(train_data, train_target)
#     target_pred = clf.predict(test_data)
#     print('第', k, '折验证分数为')
#     print(classification_report(test_target, target_pred))


# 模型验证---------------------------------------------------------------------------------------------------------------]
# 混淆矩阵
class_names = ['no-repeat', 'repeat']
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3)
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
