#coding=utf8
from numpy import *

import cal
import time

import random as r
import matplotlib.pyplot as plt



def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    username = []
    for line in fr.readlines():
        curLine = line.strip().split("    ")
        username.append(curLine[0])
#         print curLine
        l = len(curLine)
        curLine = curLine[1:l-1]
#         print curLine
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
#     print "data[0]"+str(dataMat[0])
#     print "data[1]"+str(dataMat[1])
    return username,dataMat



def loadDataSet_(fileName):
    dataMat = []
    fr = open(fileName)
    label = []
    for line in fr.readlines():
        curLine = line.strip().split(",")
#         print curLine
        label.append(curLine[-1])
        l = len(curLine)
        curLine = curLine[0:l-1]
#         print curLine
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
#     print "data[0]"+str(dataMat[0])
#     print "data[1]"+str(dataMat[1])
    return dataMat,label

'''
计算欧式距离
'''
def disEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))
        

'''
初始化k个质心的函数
'''
def randCent(dataSet,k):
    dataSet = mat(dataSet)
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        # minJ = min(dataSet[:,j])
        # rangeJ = float(max(dataSet[:,j])-minJ)
        minJ = dataSet[:, j].min(0)
        rangeJ = float(dataSet[:, j].max(0) - minJ)
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids

'''
加载PSO算法求解的初始质心
'''
def loadCent(k):
#     k=2
    if k==2: 
        center = [
                  [0.631578947368, 0.285714285714 ,0.0 ,0.0 ,0.56, 0.428571428571 ,0.1 ,0.1], 
                  [0.631578947368, 0.285714285714 ,0.25, 0.5 ,0.52 ,0.428571428571, 0.3, 0.1]
                ]



# k=3
    if k==3:
        center = [
                  [0.368421052632 ,0.285714285714, 0.25 ,0.25, 0.52 ,0.571428571429, 0.2, 0.1], 
                  [0.0526315789474 ,0.571428571429 ,0.25 ,0.5 ,0.0 ,0.857142857143 ,0.3 ,0.2  ],
                  [0.736842105263, 0.285714285714, 0.25 ,0.5, 0.76 ,0.428571428571 ,0.3 ,0.3   ]
                ]

    if k==4:
        [[0.315789473684 ,0.142857142857 ,0.0, 0.0 ,0.32, 0.285714285714 ,0.2,0.1],
         [ 0.105263157895, 0.428571428571, 0.0 ,0.5, 0.12 ,0.714285714286,0.3, 0.3],
         [ 0.736842105263 ,0.285714285714, 0.25 ,0.5 ,0.76,0.428571428571, 0.3 ,0.3,],
         [  0.736842105263, 0.285714285714, 0.25, 0.25 ,0.72, 0.714285714286, 0.0, 0.0 ]]


    center = mat(center)
    return center



def localCent_():
    centers = [
               [[13.78,14.06,0.8759,5.479,3.156,3.136,4.872,],[19.18,16.63,0.8717,6.369,3.681,3.357,6.229,],[10.79,12.93,0.8107,5.317,2.648,5.462,5.194,]],
                [[18.89,16.23,0.9008,6.227,3.769,3.639,5.966,],[10.8,12.57,0.859,4.981,2.821,4.773,5.063,],[12.88,13.5,0.8879,5.139,3.119,2.352,4.607,]],
                [[15.49,14.94,0.8724,5.757,3.371,3.412,5.228,],[18.3,15.89,0.9108,5.979,3.755,2.837,5.962,],[10.8,12.57,0.859,4.981,2.821,4.773,5.063,]],
                [[13.84,13.94,0.8955,5.324,3.379,2.259,4.805,],[11.81,13.45,0.8198,5.413,2.716,4.898,5.352,],[19.11,16.26,0.9081,6.154,3.93,2.936,6.079,]],
                [[13.45,14.02,0.8604,5.516,3.065,3.531,5.097,],[18.72,16.34,0.881,6.219,3.684,2.188,6.097,],[10.82,12.83,0.8256,5.18,2.63,4.853,5.089,]],
                [[17.63,15.98,0.8673,6.191,3.561,4.076,6.06,],[15.11,14.54,0.8986,5.579,3.462,3.128,5.18,],[11.36,13.05,0.8382,5.175,2.755,4.048,5.263,]],
                [[15.49,14.94,0.8724,5.757,3.371,3.412,5.228,],[19.13,16.31,0.9035,6.183,3.902,2.109,5.924,],[12.26,13.6,0.8333,5.408,2.833,4.756,5.36,]],
                [[12.02,13.33,0.8503,5.35,2.81,4.271,5.308,],[14.7,14.21,0.9153,5.205,3.466,1.767,4.649,],[18.96,16.2,0.9077,6.051,3.897,4.334,5.75,]],
                [[14.33,14.28,0.8831,5.504,3.199,3.328,5.224,],[11.49,13.22,0.8263,5.304,2.695,5.388,5.31,],[18.43,15.97,0.9077,5.98,3.771,2.984,5.905,]],
                [[18.36,16.52,0.8452,6.666,3.485,4.933,6.448,],[12.38,13.44,0.8609,5.219,2.989,5.472,5.045,],[14.52,14.6,0.8557,5.741,3.113,1.481,5.487,]],
                [[14.59,14.28,0.8993,5.351,3.333,4.185,4.781,],[11.35,13.12,0.8291,5.176,2.668,4.337,5.132,],[18.3,15.89,0.9108,5.979,3.755,2.837,5.962,]],
                [[14.99,14.56,0.8883,5.57,3.377,2.958,5.175,],[18.76,16.2,0.8984,6.172,3.796,3.12,6.053,],[11.56,13.31,0.8198,5.363,2.683,4.062,5.182,]],
                [[12.15,13.45,0.8443,5.417,2.837,3.638,5.338,],[17.32,15.91,0.8599,6.064,3.403,3.824,5.922,],[14.11,14.26,0.8722,5.52,3.168,2.688,5.219,]],
                [[10.91,12.8,0.8372,5.088,2.675,4.179,4.956,],[19.15,16.45,0.889,6.245,3.815,3.084,6.185,],[15.38,14.77,0.8857,5.662,3.419,1.999,5.222,]],
                [[18.85,16.17,0.9056,6.152,3.806,2.843,6.2,],[12.13,13.73,0.8081,5.394,2.745,4.825,5.22,],[14.52,14.6,0.8557,5.741,3.113,1.481,5.487,]],
                [[11.65,13.07,0.8575,5.108,2.85,5.209,5.135,],[13.02,13.76,0.8641,5.395,3.026,3.373,4.825,],[17.08,15.38,0.9079,5.832,3.683,2.956,5.484,]],
                [[12.55,13.57,0.8558,5.333,2.968,4.419,5.176,],[20.16,17.03,0.8735,6.513,3.773,1.91,6.185,],[15.03,14.77,0.8658,5.702,3.212,1.933,5.439,]],
                [[10.8,12.57,0.859,4.981,2.821,4.773,5.063,],[14.79,14.52,0.8819,5.545,3.291,2.704,5.111,],[19.94,16.92,0.8752,6.675,3.763,3.252,6.55,]],
                [[18.43,15.97,0.9077,5.98,3.771,2.984,5.905,],[12.7,13.71,0.8491,5.386,2.911,3.26,5.316,],[16.82,15.51,0.8786,6.017,3.486,4.004,5.841,]],
                [[14.01,14.29,0.8625,5.609,3.158,2.217,5.132,],[18.95,16.42,0.8829,6.248,3.755,3.368,6.148,],[10.83,12.96,0.8099,5.278,2.641,5.182,5.185,]],
                [[12.15,13.45,0.8443,5.417,2.837,3.638,5.338,],[18.81,16.29,0.8906,6.272,3.693,3.237,6.053,],[12.46,13.41,0.8706,5.236,3.017,4.987,5.147,]],
                [[11.87,13.02,0.8795,5.132,2.953,3.597,5.132,],[17.32,15.91,0.8599,6.064,3.403,3.824,5.922,],[14.29,14.09,0.905,5.291,3.337,2.699,4.825,]],
                [[17.32,15.91,0.8599,6.064,3.403,3.824,5.922,],[11.26,13.01,0.8355,5.186,2.71,5.335,5.092,],[14.29,14.09,0.905,5.291,3.337,2.699,4.825,]],
                [[18.89,16.23,0.9008,6.227,3.769,3.639,5.966,],[12.7,13.71,0.8491,5.386,2.911,3.26,5.316,],[15.05,14.68,0.8779,5.712,3.328,2.129,5.36,]],
                [[18.89,16.23,0.9008,6.227,3.769,3.639,5.966,],[17.08,15.38,0.9079,5.832,3.683,2.956,5.484,],[12.44,13.59,0.8462,5.319,2.897,4.924,5.27,]],
                [[13.5,13.85,0.8852,5.351,3.158,2.249,5.176,],[11.87,13.02,0.8795,5.132,2.953,3.597,5.132,],[18.65,16.41,0.8698,6.285,3.594,4.391,6.102,]],
                [[12.44,13.59,0.8462,5.319,2.897,4.924,5.27,],[14.86,14.67,0.8676,5.678,3.258,2.129,5.351,],[18.27,16.09,0.887,6.173,3.651,2.443,6.197,]],
                [[17.63,15.86,0.88,6.033,3.573,3.747,5.929,],[14.11,14.18,0.882,5.541,3.221,2.754,5.038,],[11.56,13.31,0.8198,5.363,2.683,4.062,5.182,]],
                [[13.5,13.85,0.8852,5.351,3.158,2.249,5.176,],[19.46,16.5,0.8985,6.113,3.892,4.308,6.009,],[12.3,13.34,0.8684,5.243,2.974,5.637,5.063,]],
                [[12.11,13.27,0.8639,5.236,2.975,4.132,5.012,],[20.71,17.23,0.8763,6.579,3.814,4.451,6.451,],[14.79,14.52,0.8819,5.545,3.291,2.704,5.111,]],
  
          ]
    center = r.sample(centers,1)[0]
    center = mat(center)
    return center


'''
K均值聚类算法
'''
def Kmean_PSO(dataSet,k,distMeas=disEclud,createCent=localCent_):
    # dataSet = mat(dataSet)
    m =  shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
#     print clusterAssment
    centroids = createCent()
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        total = 0.0
        #对每个数据找距离最近的质心
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                # print centroids[j,:]
                distJI = distMeas(centroids[j, :],dataSet[i, :])
                if distJI<minDist:
                    minDist = distJI
                    minIndex = j
            #统计f(x),即对每个簇统计距离和
            total+=minDist
            if clusterAssment[i,0]!=minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print total
        #重新调整各簇的质心的位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0],:]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment


def Kmean(dataSet,k,distMeas=disEclud,createCent=randCent):
    # dataSet = mat(dataSet)
    m =  shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
#     print clusterAssment
    centroids = createCent(dataSet,3)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        total = 0.0
        #对每个数据找距离最近的质心
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                # print centroids[j,:]
                distJI = distMeas(centroids[j, :],dataSet[i, :])
                if distJI<minDist:
                    minDist = distJI
                    minIndex = j
            #统计f(x),即对每个簇统计距离和
            total+=minDist
            if clusterAssment[i,0]!=minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print total
        #重新调整各簇的质心的位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0],:]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment


def runPSO_kmeans(filename):
    dataMat,label = loadDataSet_(filename+'.txt')
#     print dataMat
    centroids,clusterAssment = Kmean_PSO(mat(dataMat),3)

    

def runKmean(filename):
    dataMat,label = loadDataSet_(filename+'.txt')
#     print dataMat
    centroids,clusterAssment = Kmean(mat(dataMat),3)

    
def paint_iris():
    x = [i for i in range(15)]
    for i in range(len(x)):
        x[i] = x[i]+1
    y1 = [205.307947259,124.657169775,115.195027611,103.350044953,99.6785046282,98.4863804781,98.1721652356,97.9181533806,97.5879921524,97.4001520605,97.327198089,97.2464018979,97.1901252225,97.3462196942]
    y2 = [112.070560052,97.3956000608,97.327198089,97.2464018979,97.1901252225,97.3462196942]
    fig = plt.figure()  
    ax = fig.add_subplot(1,1,1)
    l = len(x)
    if len(y1)!=l:
        c = l-len(y1)
        num = y1.pop()
        for i in range(c+1):
            y1.append(num)

    if len(y2)!=l:
        c = l-len(y2)
        num = y2.pop()
        for i in range(c+1):
            y2.append(num)
    plt.xlabel("Number of iterations")
    plt.ylabel("f(x)")
    plt.xlim(1,15)
#     plt.ylim(96, 210)
    plt.plot(x,y1,color ='blue', label='kmeans_iris')
    plt.plot(x,y2,color ='red', linewidth=1.0, linestyle='--',label='PSO_kmeans_iris')
    plt.legend(loc='upper right')
    plt.show()
    
    
def paint_wine():
    x = [i for i in range(15)]
    for i in range(len(x)):
        x[i] = x[i]+1
    y1 = [31351.8232609,22534.466753,21333.7478185,20705.7502834,19554.7414943,19186.8432245,19000.1485073,19014.8728787,18994.1498755,18755.6650531,18609.2220572,18515.8678617,18489.7968411,18461.1593966,18436.9520693]
    y2 = [21170.0443788,19110.4388471,18994.3656391,19006.5684834,19014.8728787,18994.1498755,18755.6650531,18609.2220572,18515.8678617,18489.7968411,18461.1593966,18436.9520693]
    fig = plt.figure()  
    ax = fig.add_subplot(1,1,1)
    l = len(x)
    if len(y1)!=l:
        c = l-len(y1)
        num = y1.pop()
        for i in range(c+1):
            y1.append(num)

    if len(y2)!=l:
        c = l-len(y2)
        num = y2.pop()
        for i in range(c+1):
            y2.append(num)
    plt.xlabel("Number of iterations")
    plt.ylabel("f(x)")
    plt.xlim(1,15)
#     plt.ylim(96, 210)
    plt.plot(x,y1,color ='blue', label='kmeans_wine')
    plt.plot(x,y2,color ='red', linewidth=1.0, linestyle='--',label='PSO_kmeans_wine')
    
    plt.legend(loc='upper right')
    plt.show()
    
    
def paint_seeds():
    x = [i for i in range(15)]
    for i in range(len(x)):
        x[i] = x[i]+1
    y1 = [880.370209008,441.471572391,406.988848436,401.308375762,398.169093287,391.680603773,375.223081347,353.387344627,335.416695113,320.844968492,317.396875283,315.103994573,313.939414401,313.734258961]
    y2 = [394.906602047,354.567481487,337.264064069,323.402833478,317.885325699,315.103994573,313.939414401,313.734258961]
    fig = plt.figure()  
    ax = fig.add_subplot(1,1,1)
    l = len(x)
    if len(y1)!=l:
        c = l-len(y1)
        num = y1.pop()
        for i in range(c+1):
            y1.append(num)

    if len(y2)!=l:
        c = l-len(y2)
        num = y2.pop()
        for i in range(c+1):
            y2.append(num)
    plt.xlabel("Number of iterations")
    plt.ylabel("f(x)")
    plt.xlim(1,15)
#     plt.ylim(96, 210)
    plt.plot(x,y1,color ='blue',label='kmeans_seeds')
    plt.plot(x,y2,color ='red', linewidth=1.0, linestyle='--',label='PSO_kmeans_seeds')
    
    plt.legend(loc='upper right')
    plt.show()



if __name__=="__main__":
# # 
#     runKmean("seeds")
#     print "----------------------------"
#     runPSO_kmeans('seeds')

    paint_iris()
    paint_wine()
    paint_seeds()
    
    