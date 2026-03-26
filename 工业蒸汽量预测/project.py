import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge  # 导入岭回归模型
from sklearn.metrics import mean_squared_error  # 导入均方误差损失函数
from sklearn import preprocessing  # 数据处理包
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 多重共线性分析函数
from sklearn.decomposition import PCA  # 导入PCA模型
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.neighbors import KNeighborsRegressor  # K临近回归
from sklearn.tree import DecisionTreeRegressor  # 决策树回归
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归
from sklearn.svm import SVR  # 支持向量回归
from sklearn.linear_model import Lasso  # Lasso回归
from sklearn.linear_model import ElasticNet  # ElasticNet回归
from sklearn.ensemble import GradientBoostingRegressor  # GBDT
from sklearn.ensemble import AdaBoostRegressor  # AdaBoost
from sklearn.model_selection import train_test_split  # 切分数据
from sklearn.model_selection import KFold  # K折交叉验证
from sklearn.model_selection import RepeatedKFold  # 重复K折交叉验证
from sklearn.model_selection import LeaveOneOut  # 留一法交叉验证
from sklearn.model_selection import LeavePOut  # 留p法交叉验证
from sklearn.model_selection import GridSearchCV  # 网格搜索参数优化
from sklearn.model_selection import RandomizedSearchCV  # 随机参数优化
from sklearn.model_selection import learning_curve  # 学习曲线
from sklearn.model_selection import validation_curve  # 验证曲线
from sklearn.model_selection import ShuffleSplit  # 随机分为训练集与检测集
from sklearn.linear_model import SGDRegressor  # 随机梯度下降回归
from sklearn.model_selection import cross_val_score  # 生成重复训练模型的分数
import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings ("ignore") 是一个用于控制警告输出的函数调用。 当你在 Python 代码中使用这行代码时，
# 它会告诉 Python 解释器忽略所有的警告消息，从而防止这些警告消息干扰你的程序执行。


# 数据探索---------------------------------------------------------------------------------------------------------------
# 读取数据
train_data_file = "zhengqi_train.txt"
test_data_file = "zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep="\t", encoding="utf-8")
test_data = pd.read_csv(test_data_file, sep="\t", encoding="utf-8")


# 变量识别与缺失值处理

# 查看基本信息
# train_data.info()
# test_data.info()

# 查看统计信息
# print(train_data.describe())
# print(test_data.describe())

# 查看数据字段信息
# print(train_data.head())
# print(test_data.head())


# 数据分析与异常值处理
# 单变量分析
# 绘制箱型图
# plt.figure(figsize=(10, 10))
# plt.boxplot(x=train_data.values, labels=train_data.columns)
# plt.hlines([-7.5, 7.5], 0, 40, colors='r')
# plt.show()
# 发现V9出现个别异常值，考虑移除
train_data = train_data[train_data['V9'] > -7.5]
test_data = test_data[test_data['V9'] > -7.5]

# 获取异常数据并画图
# def find_outliers(model, X, y, sigma=3):
#     try:
#         y_pred = pd.Series(model.predict(X), index=y.index)
#     except:
#         model.fit(X, y)
#         y_pred = pd.Series(model.predict(X), index=y.index)
#     resid = y - y_pred
#     mean_resid = resid.mean()
#     std_resid = resid.std()
#     z = (resid - mean_resid)/std_resid
#     outliers = z[abs(z)>sigma].index
#     print("R^2=", model.score(X, y))
#     print("mse=", mean_squared_error)
#     print("------------------------------------")
#     print("mean of residuals:", mean_resid)
#     print("std of residuals:", std_resid)
#     print("------------------------------------")
#     print(len(outliers), "outliers:",)
#     print(outliers.tolist)
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.plot(y, y_pred, ".")
#     plt.plot(y.loc[outliers], y_pred.loc[outliers], "ro")
#     plt.legend(["accepted", "outlier"])
#     plt.xlabel("y")
#     plt.ylabel("y_pred")
#     plt.subplot(1, 3, 2)
#     plt.plot(y, y - y_pred, ".")
#     plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], "ro")
#     plt.legend(["accepted", "outlier"])
#     plt.xlabel("y")
#     plt.ylabel("y - y_pred")
#     ax_133 = plt.subplot(1, 3, 3)
#     z.plot.hist(bins=50, ax=ax_133)
#     z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
#     plt.legend(["accepted", "outlier"])
#     plt.xlabel("z")
#     plt.savefig("outliers.png")
#     return outliers
# X_train = train_data.iloc[:, 0:-1]
# y_train = train_data.iloc[:, -1]
# outliers = find_outliers(Ridge(), X_train, y_train)

# 绘制直方图与Q-Q图
# train_cols = 8
# train_rows = len(train_data.columns)
# plt.figure(figsize=(4*train_cols, 4*train_rows))
# i = 0
# for col in train_data.columns:
#     i += 1
#     plt.subplot(int(train_rows/(train_cols/2))+1, train_cols, i)
#     sns.distplot(train_data[col], fit=stats.norm)
#     i += 1
#     plt.subplot(int(train_rows/(train_cols/2))+1, train_cols, i)
#     stats.probplot(train_data[col], plot=plt)
# plt.tight_layout()
# plt.show()

# 绘制KDE分布图
# dist_cols = 6
# dist_rows = len(test_data.columns)
# plt.figure(figsize=(4*dist_cols, 4*dist_rows))
# i = 1
# for col in test_data.columns:
#     plt.subplot(int(dist_rows/dist_cols)+1, dist_cols, i)
#     sns.kdeplot(train_data[col], color="red", shade=True)
#     sns.kdeplot(test_data[col], color="blue", shade=True)
#     plt.legend(["train", "test"])
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     i += 1
# plt.tight_layout()
# plt.show()
# 由图可以去除属性V5, V9, V11, V17, V22, V28
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
train_data.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)
test_data.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)

# 绘制线性回归关系图
# f_cols = 6
# f_rows = len(test_data.columns)
# plt.figure(figsize=(4*f_cols, 4*f_rows))
# i = 0
# for col in test_data.columns:
#     i += 1
#     ax = plt.subplot(int(f_rows/(f_cols/2))+1, f_cols, i)
#     sns.regplot(x=col, y='target', data=train_data, scatter_kws={'marker': '.', 's': 3, "alpha": 0.3},
#                 line_kws={'color': 'k'})
#     plt.xlabel(col)
#     plt.ylabel('target')
#     i += 1
#     plt.subplot(int(f_rows/(f_cols/2))+1, f_cols, i)
#     sns.distplot(train_data[col].dropna())
#     plt.xlabel(col)
# plt.tight_layout()
# plt.show()

# 多变量分析
# 计算相关性系数
train_corr = train_data.corr()
# print("train_corr的相关系数矩阵：")
# print(train_corr)

# 绘制相关性热力图
# plt.figure(figsize=(20, 16))
# sns.heatmap(train_corr, vmax=.8, square=True, annot=True)
# plt.show()

# 根据相关系数筛选特征变量
# k = 10  # 找出与target变量的相关系数最大的k个
# cols = train_corr.nlargest(k, 'target')['target'].index
# plt.figure(figsize=(10, 10))
# sns.heatmap(train_data[cols].corr(), annot=True, square=True)
# plt.show()
threshold = 0.1  # 找出与target变量的相关系数大于0.1的特征变量
top_corr_features = train_corr.index[abs(train_corr['target']) > threshold]
# plt.figure(figsize=(10, 10))
# sns.heatmap(train_data[top_corr_features].corr(), annot=True, square=True)
# plt.show()
drop_col = train_corr.index[abs(train_corr['target']) < threshold]  # 用相关系数阈值移除相关特征
# print("用相关系数移除的特征：")
# print(drop_col)
train_data.drop(drop_col, inplace=True, axis=1)
test_data.drop(drop_col, inplace=True, axis=1)


# 变量转换
# Box-Cox变换
# 将train_data与test_data结合到一起
# train_x = train_data.drop(['target'], axis=1)
# data_all = pd.concat([train_x, test_data])
# print("Box-Cox转换前的data_all数据:\n")
# print(data_all.describe())
# cols_numeric = list(data_all.columns)
# 将data_all进行归一化
# def scale_minmax(col):
#     return (col-col.min())/(col.max()-col.min())
# data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis=0)
# 开始Box-Cox转换
# cols_numeric_left = cols_numeric[0:4]
# cols_numeric_right = cols_numeric[4:]
# data_all = pd.concat([data_all, train_data['target']], axis=1)
# f_cols = 6
# f_rows = len(cols_numeric_left)
# plt.figure(figsize=(4*f_cols, 4*f_rows))
# i = 0
# for var in cols_numeric:
#     dat = data_all[[var, 'target']].dropna()
#     i += 1
#     plt.subplot(f_rows, f_cols, i)
#     sns.distplot(dat[var], fit=stats.norm)
#     plt.title(var+"original")
#     i += 1
#     plt.subplot(f_rows, f_cols, i)
#     stats.probplot(dat[var], plot=plt)
#     plt.title("skew=" + '{:.4f}'.format(stats.skew(dat[var])))
#     i += 1
#     plt.subplot(f_rows, f_cols, i)
#     plt.plot(dat[var], dat['target'], '.', alpha=0.5)
#     plt.title('corr=' + '{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))
#     i += 1
#     plt.subplot(f_rows, f_cols, i)
#     trans_var, lambda_var = stats.boxcox(dat[var].dropna()+1)
#     trans_var = scale_minmax(trans_var)
#     sns.distplot(trans_var, fit=stats.norm)
#     plt.title(var+'transformed')
#     i += 1
#     plt.subplot(f_rows, f_cols, i)
#     stats.probplot(trans_var, plot=plt)
#     plt.title('skew='+'{:.4f}'.format(stats.skew(trans_var)))
#     i += 1
#     plt.subplot(f_rows, f_cols, i)
#     plt.plot(trans_var, dat['target'], '.', alpha=0.5)
#     plt.title('corr=' + '{:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))
# plt.show()

# 最大值和最小值的归一化
features_columns = [col for col in train_data.columns if col not in ['target']]
min_max_scaler = preprocessing.MinMaxScaler()
train_data_scaler = min_max_scaler.fit_transform(train_data[features_columns])
test_data_scaler = min_max_scaler.fit_transform(test_data[features_columns])
train_data_scaler = pd.DataFrame(train_data_scaler)
train_data_scaler.columns = features_columns
test_data_scaler = pd.DataFrame(test_data_scaler)
test_data_scaler.columns = features_columns
train_data_scaler['target'] = train_data['target']
# print("最大值和最小值归一化后的训练集与测试集:")
# print(train_data_scaler.describe())
# print(test_data_scaler.describe())


# 特征工程---------------------------------------------------------------------------------------------------------------
# 多重共线性分析
X = np.matrix(train_data_scaler)
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
# print("多重共线性分析结果:")
# print(VIF_list)

# PCA处理
pca = PCA(n_components=16)  # 保留16个主成分
new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:, 0:-1])
new_test_pca_16 = pca.transform(test_data_scaler)
new_train_pca_16 = pd.DataFrame(new_train_pca_16)
new_test_pca_16 = pd.DataFrame(new_test_pca_16)
new_train_pca_16['target'] = train_data_scaler['target']
# print("PCA处理后的训练集与测试集:")
# print(new_train_pca_16.describe())
# print(new_test_pca_16.describe())


# 模型训练---------------------------------------------------------------------------------------------------------------
# 切分数据
new_train_pca_16 = new_train_pca_16.fillna(0)
train = new_train_pca_16[new_train_pca_16.columns]
train.drop(['target'], inplace=True, axis=1)
target = new_train_pca_16['target']
train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)

# 多元线性回归
# clf = LinearRegression()
# clf.fit(train_data, train_target)
# score = mean_squared_error(test_target, clf.predict(test_data))
# print('LinearRegression:', score)

# K临近回归
# clf = KNeighborsRegressor(n_neighbors=8)
# clf.fit(train_data, train_target)
# score = mean_squared_error(test_target, clf.predict(test_data))
# print('KNeighborsRegressor:', score)

# 随机森林回归
# clf = RandomForestRegressor(n_estimators=20)
# clf.fit(train_data, train_target)
# score = mean_squared_error(test_target, clf.predict(test_data))
# print('RandomForestRegressor:', score)

# 决策树回归
# clf = DecisionTreeRegressor()
# clf.fit(train_data, train_target)
# score = mean_squared_error(test_target, clf.predict(test_data))
# print('DecisionTreeRegressor:', score)

# SVR回归
# clf = SVR()
# clf.fit(train_data, train_target)
# score = mean_squared_error(test_target, clf.predict(test_data))
# print('SVR:', score)


# 模型验证---------------------------------------------------------------------------------------------------------------
# 模型评估
# 简单交叉验证与模型训练代码相同

# K折交叉验证
# kf = KFold(n_splits=5)
# for k, (train_index, test_index) in enumerate(kf.split(train)):
#     train_data, test_data, train_target, test_target = train.values[train_index], train.values[test_index], \
#         target[train_index], target[test_index]
#     clf = LinearRegression()
#     clf.fit(train_data, train_target)
#     score_train = mean_squared_error(train_target, clf.predict(train_data))
#     score_test = mean_squared_error(test_target, clf.predict(test_data))
#     print(k, "折", "LinearRegression train MSE:", score_train)
#     print(k, "折", "LinearRegression test MSE:", score_test, "\n")

# 留一法交叉验证
# loo = LeaveOneOut()
# for k, (train_index, test_index) in enumerate(loo.split(train)):
#     train_data, test_data, train_target, test_target = train.values[train_index], train.values[test_index], \
#         target.values[train_index], target.values[test_index]
#     clf = LinearRegression()
#     clf.fit(train_data, train_target)
#     score_train = mean_squared_error(train_target, clf.predict(train_data))
#     score_test = mean_squared_error(test_target, clf.predict(test_data))
#     print(k, "个", "LinearRegression train MSE:", score_train)
#     print(k, "个", "LinearRegression test MSE:", score_test, "\n")
#     if k >= 9:
#         break

# 留p法交叉验证
# lpo = LeavePOut(p=10)
# for k, (train_index, test_index) in enumerate(lpo.split(train)):
#     train_data, test_data, train_target, test_target = train.values[train_index], train.values[test_index], \
#         target.values[train_index], target.values[test_index]
#     clf = LinearRegression()
#     clf.fit(train_data, train_target)
#     score_train = mean_squared_error(train_target, clf.predict(train_data))
#     score_test = mean_squared_error(test_target, clf.predict(test_data))
#     print(k, "10个", "LinearRegression train MSE:", score_train)
#     print(k, "10个", "LinearRegression test MSE:", score_test)
#     if k >= 9:
#         break


# 模型调参
# 网格搜索
# randomForestRegression = RandomForestRegressor()
# parameters = {'n_estimators': [50, 100, 200], "max_depth": [1, 2, 3]}
# clf = GridSearchCV(randomForestRegression, parameters, cv=5)
# clf.fit(train_data, train_target)
# score_test = mean_squared_error(test_target, clf.predict(test_data))
# print("RandomForestRegression GridSearchCV test MSE:", score_test)

# 随机参数优化
# randomForestRegression = RandomForestRegressor()
# parameters = {'n_estimators': [50, 100, 200, 300], 'max_depth': [1, 2, 3, 4, 5]}
# clf = RandomizedSearchCV(randomForestRegression, parameters, cv=5)
# clf.fit(train_data, train_target)
# score_test = mean_squared_error(test_target, clf.predict(test_data))
# print("RandomForestRegression RandomizedSearchCV test MSE:", score_test)


# 学习曲线
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
#                                                             train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
#     alpha=0.1, color='r')
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
#                      color='g')
#     plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation score")
#     plt.legend()
#     plt.show()
#     return plt
# X = new_train_pca_16[new_test_pca_16.columns].values
# y = new_train_pca_16['target'].values
# title = "LinearRegression"
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# estimator = LinearRegression()
# plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=-1)


# 验证曲线
# X = new_train_pca_16[new_test_pca_16.columns].values
# y = new_train_pca_16['target'].values
# param_range = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
# train_scores, test_scores = validation_curve(SGDRegressor(), X, y, param_name="alpha", param_range=param_range,
#                                              cv=10, scoring='r2', n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.title("Validation Curve with LinearRegression")
# plt.xlabel("alpha")
# plt.ylabel("Score")
# plt.ylim([0, 1])
# plt.semilogx(param_range, train_scores_mean, label="Training score", color='r')
# plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
#                  color='r')
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color='g')
# plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
#                  color='g')
# plt.legend()
# plt.show()


# 模型融合---------------------------------------------------------------------------------------------------------------
# 本部分将模型训练与模型验证的所有内容结合到一起，用K折交叉验证与网格搜索训练模型
# Average融合策略
# 利用train_model函数训练模型并评价模型
# def train_model(model, X, y, param_grid, splits=5, repeats=5):
#     rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
#     if len(param_grid) > 0:
#         gsearch = GridSearchCV(model, param_grid, cv=rkfold, scoring="neg_mean_squared_error")
#         gsearch.fit(X, y)
#         model = gsearch.best_estimator_
#         best_idx = gsearch.best_index_
#         grid_results = pd.DataFrame(gsearch.cv_results_)
#         cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
#         cv_std = grid_results.loc[best_idx, 'std_test_score']
#     else:
#         grid_results = []
#         cv_results = cross_val_score(model, X, y, scoring='neg_mean_square_error', cv=rkfold)
#         cv_mean = abs(np.mean(cv_results))
#         cv_std = np.std(cv_results)
#     cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})
#     y_pred = model.predict(X)
#     print('------------------------')
#     print(model)
#     print('------------------------')
#     print('score=', model.score(X, y))
#     print('mse=', mean_squared_error(y, y_pred))
#     print('cross-val:mean=', cv_mean, ', std=', cv_std)
#     return model, cv_score, grid_results
# opt_models = dict()
# score_models = pd.DataFrame(columns=['mean', 'std'])
# X = new_train_pca_16[new_test_pca_16.columns]
# y = new_train_pca_16['target']

# 岭回归
# model = 'Ridge'
# opt_models[model] = Ridge()
# algh_range = np.arange(0.25, 6, 0.25)
# param_grid = {'alpha': algh_range}
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], X, y, param_grid)
# cv_score.name = model
# score_models = score_models._append(cv_score)

# LASSO回归
# model = 'Lasso'
# opt_models[model] = Lasso()
# algh_range = np.arange(1e-4, 1e-3, 4e-5)
# param_grid = {'alpha': algh_range}
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], X, y, param_grid)
# cv_score.name = model
# score_models = score_models._append(cv_score)

# ElasticNet回归
# model = 'ElasticNet'
# opt_models[model] = ElasticNet()
# param_grid = {'alpha': np.arange(1e-4, 1e-3, 1e-4), 'l1_ratio': np.arange(0.1, 1, 0.1), 'max_iter': [100000]}
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], X, y, param_grid, repeats=1)
# cv_score.name = model
# score_models = score_models._append(cv_score)

# SVR回归
# model = 'SVR'
# opt_models[model] = SVR()
# crange = np.arange(0.1, 1.0, 0.1)
# param_grid = {'C': crange, 'max_iter': [1000]}
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], X, y, param_grid)
# cv_score.name = model
# score_models = score_models._append(cv_score)

# K近邻
# model = 'KNeighbors'
# opt_models[model] = KNeighborsRegressor()
# param_grid = {'n_neighbors': np.arange(3, 11, 1)}
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], X, y, param_grid, repeats=1)
# cv_score.name = model
# score_models = score_models._append(cv_score)

# 随机森林
# model = 'RandomForest'
# opt_models[model] = RandomForestRegressor()
# param_grid = {'n_estimators': [100, 150, 200], 'max_features': [8, 12, 16, 20, 24], 'min_samples_split': [2, 4, 6]}
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], X, y, param_grid, repeats=1)
# cv_score.name = model
# score_models = score_models._append(cv_score)
# print(score_models)

# GBDT
# model = 'GradientBoosting'
# opt_models[model] = GradientBoostingRegressor()
# param_grid = {'n_estimators': [150, 250, 350], 'max_depth': [1, 2, 3], 'min_samples_split': [5, 6, 7]}
# opt_models[model], cv_score, grid_results = train_model(opt_models[model], X, y, param_grid, repeats=1)

# Average融合
# def model_predict(test_data, test_y):
#     i = 0
#     y_predict_total = np.zeros((test_data.shape[0], ))
#     for model in opt_models.keys():
#         if model != 'LinearSVR' and model != 'KNeighbors':
#             y_predict = opt_models[model].predict(test_data)
#             y_predict_total += y_predict
#             i += 1
#             print('{}_mse'.format(model), mean_squared_error(test_y, y_predict))
#     y_predict_mean = np.round(y_predict_total/i, 3)
#     print('mean_mse:', mean_squared_error(test_y, y_predict_mean))
#     return y_predict_mean
# model_predict(test_data, test_target)


# Stacking融合策略
# 利用stacking_reg训练模型并评价模型
def stacking_reg(clf, train_X, train_y, test_x, clf_name, kf):
    train = np.zeros((train_X.shape[0], 1))  # train用来存放模型输入train_X的predict结果
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((5, test_x.shape[0], 1))
    cv_score = []
    for i, (train_index, test_index) in enumerate(kf.split(train_X)):
        tr_x = train_X[train_index]
        tr_y = train_y[train_index]
        te_x = train_X[test_index]
        te_y = train_y[test_index]
        if clf_name in ['rf', 'ada', 'gb', 'et', 'lr', 'lsvc', 'knn']:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_score.append(mean_squared_error(te_y, pre))
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list" % clf_name, cv_score)
    print("%s_score_mean" % clf_name, np.mean(cv_score))
    return train.reshape(-1, 1), test.reshape(-1, 1)

# 随机森林
def rf_reg(x_train, y_train, x_valid, kf):
    RandomForest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1,
                                         max_features='auto', verbose=1)
    rf_train, rf_test = stacking_reg(RandomForest, x_train, y_train, x_valid, 'rf', kf)
    return rf_train, rf_test, 'rf_reg'

# adaboost
def ada_reg(x_train, y_train, x_valid, kf):
    adaboost = AdaBoostRegressor(n_estimators=30, learning_rate=0.01)
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid, 'ada', kf)
    return ada_train, ada_test, 'ada_rag'

# GBDT
def gb_reg(x_train, y_train, x_valid, kf):
    gbdt = GradientBoostingRegressor(learning_rate=0.04, n_estimators=100, subsample=0.8, max_depth=5, verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid, 'gb', kf)
    return gbdt_train, gbdt_test, 'gb_reg'

# LinearRegression
def lr_reg(x_train, y_train, x_valid, kf):
    lr_reg = LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, 'lr', kf)
    return lr_train, lr_test, 'lr_reg'

# Stacking融合
# def stacking_pred(x_train, y_train, x_valid, kf, clf_list, clf_fin):
#

