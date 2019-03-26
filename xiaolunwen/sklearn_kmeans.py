# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
import operator
import random

# data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
# print type(data)

def loadData():
#     fp = open("iris.txt",'r')
#     fp = open("seeds.txt",'r')
    fp = open("wine.txt",'r')
    D = []
    labels = []
    lines= fp.readlines()
    random.shuffle(lines)
    for line in lines:
        data = line.strip()
        datas = data.split(",")
        label = datas.pop()
        D.append(datas)
        labels.append(label)    
#     print D
    return D,labels

def getCorrect(label_pred_list,label):
    m = {}
    for i in range(len(label_pred_list)):
        key = str(label_pred_list[i])+":"+str(label[i])
        if key in m.keys():
            m[key] = m[key]+1
        else:
            m[key]=1
#     print m
    list_res = sorted(m.iteritems(), key = operator.itemgetter(1), reverse = True)
    #计算错误率
    k=0
    i=0
    for item in list_res:
        i+=1
        if i==4:
            break;
        k+=int(item[1])
    num = len(label)
    print "聚类正确的个数为:"+str(k)
    print "聚类正确率为:"+str(float(k)/num)

D,label = loadData()
data = np.array(D)
#假如我要构造一个聚类数为3的聚类器
estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

print label_pred
print centroids
print inertia

#获取聚类正确率
getCorrect(label_pred.tolist(), label)
