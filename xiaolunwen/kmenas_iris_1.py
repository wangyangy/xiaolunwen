#-*-coding:utf-8 -*-
import math
from collections import defaultdict
import numpy as np
dataname = "iris.txt"
def loadIRISdata(filename):
    data = []
    with open(filename, mode="r") as rf:
        for line in rf:
            if line == '\n':
                continue
            line = line.split(",")[0:4]
#             print len(line)
            data.append(list(map(float, line)))
    return data
 
def generateCenters(data):
    '''求解初始聚类中心'''
    centers = []
    '''已知维度为4'''
    '''分三类，取第0，50，100的三个向量作为分界'''
    centers.append(data[0])
    centers.append(data[50])
    centers.append(data[100])
    return centers
 
def distance(a ,b):
    '''欧式距离'''
    sum = 0
    for i in range(4):
        sq = (a[i]-b[i])*(a[i]-b[i])
        sum += sq
    return math.sqrt(sum)
 
def point_avg(points):
    '''对维度求平均值'''
    new_center = []
    for i in range(4):
        sum = 0
        for p in points:
            sum += p[i]
        new_center.append(float("%.8f" % (sum/float(len(points)))))
    return new_center
 
def updataCenters(data, assigments):
    new_means = defaultdict(list)
    centers = []
    for assigment, point in zip(assigments, data):
        new_means[assigment].append(point)
        '''将同一类的数据进行整合'''
    for i in range(3):
        points = new_means[i]
        centers.append(point_avg(points))
    return centers
 
def assignment(data, centers):
    assignments = []
    '''对应位置显示对应类群'''
    for point in data:
        '''遍历所有数据'''
        shortest = float('inf')
        shortestindex = 0
        for i in range(3):
            '''遍历三个中心向量，与哪个类中心欧氏距离最短就将其归为哪类'''
            value = distance(point, centers[i])
            if value < shortest:
                shortest = value
                shortestindex = i
        assignments.append(shortestindex)
    return assignments
 
def kmeans(data):
    k_data = generateCenters(data)
    assigments = assignment(data, k_data)
    old_assigments = None
    while assigments != old_assigments:
        new_centers = updataCenters(data, assigments)
        old_assigments = assigments
        assigments = assignment(data, new_centers)
    result = list(zip(assigments, data))
    return result
 
def acc(result):
    sum = 0
    all = 0
    for i in range(50):
        if result[i][0] == 0:
            sum += 1
        all += 1
    for i in range(50):
        if result[i+50][0] == 1:
            sum += 1
        all += 1
    for i in range(50):
        if result[i+100][0] == 2:
            sum += 1
        all += 1
    print('sum:', sum, 'all:', all)
    return sum, all
 
if __name__ == "__main__":
    data = loadIRISdata(dataname)
    result = kmeans(data)
    for i in range(3):
        tag = 0
        print('\n')
        print("第%d类数据有：" % (i+1))
        for tuple in range(len(result)):
            if(result[tuple][0] == i):
                print tuple
                tag += 1
            if tag > 20 :
                print('\n')
                tag = 0
    #print(result)
    print('\n')
    sum, all = acc(result)
    print('c-means准确度为:%2f%%' % ((sum/all)*100))
