#coding=utf8
from numpy import *
import cal
import time
import random as rd



# print random.rand(0,30)

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

    centers =[ 
            [[120.640625,56.05643562,0.048282013,-0.577985533,6.816053512,29.41877787,5.133233118,27.39578076,],[106.625,42.39748666,0.280169597,0.206694043,5.544314381,29.62971949,5.63180563,32.09661617,]],
            [[95.8125,39.2517416,0.894679694,1.564413195,2.469899666,19.43955378,8.875037583,84.46929592,],[105.6015625,50.15481343,0.380509034,0.051593251,3.433110368,22.07136587,8.735448902,82.598087,]],
            ]
    center = centers[rd.randint(0,1)]    
    center = mat(center)
#     print center
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
#         print total
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
    centroids = createCent(dataSet,k)
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
#         print total
        #重新调整各簇的质心的位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0],:]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment


def runPSO_kmeans(filename,k):
    dataMat,label = loadDataSet_(filename+'.txt')
#     print dataMat
    centroids,clusterAssment = Kmean_PSO(mat(dataMat),k)
    #聚类中心很大可能不是数据集中的点，可能只是一个随机的值
#     print centroids
#     print clusterAssment
#     print type(clusterAssment)
    res = clusterAssment.tolist()
#     print res
    fp = open(filename+"_res_PSOkmeans.txt",'w')
    for i in res:
        fp.write(str(i[0])+"    "+str(i[1])+"\n")
    fp.flush()
    fp.close()
    time.sleep(1)
    r = cal.cal(filename,"PSO_kmeans")
    return r
    

def runKmean(filename,k):
    dataMat,label = loadDataSet_(filename+'.txt')
#     print dataMat
    centroids,clusterAssment = Kmean(mat(dataMat),k)
    #聚类中心很大可能不是数据集中的点，可能只是一个随机的值
#     print centroids
#     print clusterAssment
#     print type(clusterAssment)
    res = clusterAssment.tolist()
#     print res
    fp = open(filename+"_res_kmeans.txt",'w')
    for i in res:
        fp.write(str(i[0])+"    "+str(i[1])+"\n")
    fp.flush()
    fp.close()
    time.sleep(1)
    r = cal.cal(filename,"kmeans")
    return r
    


if __name__=="__main__":
#     print "------------------------------原始kmeans算法------------------------------"
#     r=0.0
#     for i in range(30):
#         r += runKmean("iris")
#     print "------------------------------改进的PSO-kmeans算法------------------------------"
#     for i in range(30):
#         runPSO_kmeans('iris')
#     cen = localCent_()
#     print cen
    for i in range(20):
        start = time.time()
        runKmean("newData//magic04",2)
        end = time.time()
        print "******kmeans花费的时间为"+str(end-start)
#         start = time.time()
#         runPSO_kmeans('newData//magic04',2)
#         end = time.time()
#         print "******PSO_kmeans花费的时间为"+str(end-start)
        print "---------------------------------------"
        time.sleep(5)
        
#     runKmean("smallData//iris")
#     runPSO_kmeans('smallData//iris')
    
#     #连续运行可能会报错，不知道为啥
#     r1 = 0.0
#     r2 = 0.0
#     #进行30次试验来求平均值
#     for i in range(30):
#         r1 += runKmean("iris")
#         r2 += runPSO_kmeans('iris')
#     print "30次试验的平均每结果为"
#     print r1/30
#     print r2/30
    
    