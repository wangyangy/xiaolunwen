#-*-coding:utf-8 -*-
"""
功能：K均值聚类
说明：人为设置函数模型为2类
作者：唐天泽
博客：http://blog.csdn.net/u010837794/article/details/76596063
日期：2017-08-04
"""

"""
导入项目所需的包
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.datasets import load_iris


from sklearn.cluster import KMeans

# 使用交叉验证的方法，把数据集分为训练集合测试集
from sklearn.model_selection import train_test_split

# 加载数据集
def load_data():
    iris = load_iris()
#     iris = datasets.diabetes()
    """展示数据集的形状
       diabetes.data.shape, diabetes.target.shape
    """

    # 将数据集拆分为训练集和测试集 
    X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.10, random_state=0)
    return X_train, X_test, y_train, y_test
# 使用KMeans考察线性分类KMeans的预测能力
def test_KMeans(X_train,X_test,y_train,y_test):

    # 选择模型,把数据交给模型训练
    y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(X_train)

    """绘图"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X_train[:, 2], X_train[:, 3], c=y_pred)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("K-means")
    ax.legend(framealpha=0.5)
    plt.show()
    return
if __name__=="__main__":
    X_train,X_test,y_train,y_test=load_data() # 生成用于分类的数据集
    test_KMeans(X_train,X_test,y_train,y_test) # 调用 test_KMeans
