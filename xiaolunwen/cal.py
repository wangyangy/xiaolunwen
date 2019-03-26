#-*-coding:utf-8 -*-
from numpy import *
from math import sqrt
import operator;

#重新计算算法分类的准确星,
import sys

 
def loadData(fileName):
    res = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split(',')
        res.append(curline[-1])
    return res

def loadData_(fileName):
    res = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split('    ')
        res.append(curline[0])
    return res


'''需要传入存储结果的文件的名称,这个计算的有问题，
以数据中的标签为key，统计对应的      label1-label2,可能会有多种类型，这时候统计这几种类型中数量最多的为当前的聚类的标签（少数服从多数）
在计算错误率的时候不管不同类别的是否聚为一类，只管同一类别的聚类出错的部分来计算错误率
g--1.0:10419
h--1.0:4761
h--0.0:1927
g--0.0:1913
err = (1927+1913)/19020
rig = (10419+4761)/19020
'''
def cal(filename,algorthm):
    res = loadData(filename+".txt")
    if "kmeans"==algorthm:
        res_ = loadData_(filename+"_res_kmeans.txt")
    elif "PSO_kmeans"==algorthm:
        res_ = loadData_(filename+"_res_PSOkmeans.txt")
        
    num = len(res)
#     print num
    label_ = set(res)
    #统计每种匹配的个数
    map_ = {}
    for i in range(len(res)):
        s=""
        s+=res[i]+"--"+res_[i]
        if s not in map_.keys():
            map_[s] = 1
        else:
            map_[s] += 1
    for key in map_.keys():
        print str(key)+":"+str(map_.get(key))
    Map = {}
#     Map.setdefault("",{})
    for item in label_:
        item = str(item)
        for key in map_.keys():
            key = str(key)
            ind = key.find(item)
            if ind==0:
                if item not in Map.keys():
                    submap = {}
                else:
                    submap = Map[item]
                jianzhidui = {key:map_.get(key)}
                submap.update(jianzhidui)
                Map[item] = submap
    #获取真正的对应类别
    lab_lab_num = {}
    for key in Map.keys():
        submap = Map.get(key)
        list_res = sorted(submap.iteritems(), key = operator.itemgetter(1), reverse = True)
        lab_lab_num.update({list_res[0][0]:list_res[0][1]})
#     list_res = sorted(map_.iteritems(), key = operator.itemgetter(1), reverse = True)
#     for item in list_res:
#         print item
    #计算错误率
    rightK = 0
    errorK = 0
    for key in map_.keys():
        if key not in lab_lab_num.keys():
            errorK += map_.get(key)
        else:
            rightK += map_.get(key)
        pass
    print "聚类正确的个数为:"+str(rightK)
    print "聚类正确率为:"+str(float(rightK)/num)
    r = float(rightK)/num
    return r


def cal_(filename,algorthm):
    res = loadData(filename+".txt")
    if "kmeans"==algorthm:
        res_ = loadData_(filename+"_res_kmeans.txt")
    elif "PSO_kmeans"==algorthm:
        res_ = loadData_(filename+"_res_PSOkmeans.txt")
        
    num = len(res)
#     print num
    map = {}
    for i in range(len(res)):
        s=""
        s+=res[i]+"--"+res_[i]
        if s not in map.keys():
            map[s] = 1
        else:
            map[s] += 1
#     for key in map.keys():
#         print key,str(map[key])
    list_res = sorted(map.iteritems(), key = operator.itemgetter(1), reverse = True)
#     for item in list_res:
#         print item
    #计算错误率
    k=0
    i=0
    for item in list_res:
        i+=1
        if i==4:
            break;
        k+=int(item[1])
    print "聚类正确的个数为:"+str(k)
    print "聚类正确率为:"+str(float(k)/num)
    r = float(k)/num
    return r

    



if __name__ == "__main__":
    cal()
    
    
    
   
    