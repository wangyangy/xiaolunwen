#coding=utf8
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans
from sklearn import datasets  

from sklearn.datasets import load_wine
wine = load_wine()

X = wine.data[:, 0:13] ##表示我们只取特征空间中的后两个维度
print(X.shape)
#绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c = "red", marker='o', label='see')  
plt.xlabel('')  
plt.ylabel('')  
plt.legend(loc=2)  
plt.show()



estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[label_pred == 0]
print x0
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 12], x0[:, 13], c = "red", marker='o', label='label0')  
plt.scatter(x1[:, 12], x1[:, 13], c = "green", marker='*', label='label1')  
plt.scatter(x2[:, 12], x2[:, 13], c = "blue", marker='+', label='label2')  
plt.xlabel('')  
plt.ylabel('')  
plt.legend(loc=2)  
plt.show()  