# -*- coding: utf-8 -*-
import random
import time
import copy
import numpy as np

#0,9都可以产生
# print(random.randint(0,9))
# date  = time.time()
# print str(date)
# d = time.localtime()-1000
# print str(time.strftime("%Y-%m-%d %H:%M:%S", d))


time_stamp = time.time()               # 时间戳
print str(time_stamp)
#后面减去的是秒数
local_time = time.localtime(time_stamp-100)# 时间戳转struct_time类型的本地时间
t = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
print str(t)
t = time.strptime(t, "%Y-%m-%d %H:%M:%S")
timeStamp = int(time.mktime(t))
print timeStamp


a = ['asd','adsdgr','345',1,2,3]
subborwsers = random.sample(a, 3)
print subborwsers

datas={}
datas['0']=[]
datas['0'].append([1,2,3])
datas['0'].append([5,6])
print datas['0']
print len(datas['0'])


# fp = open("data\\kmeansk_4.txt",'r')
# a1=0
# a2=0
# a3=0
# a4=0
# for line in fp.readlines():
#     line = line.strip()
#     data = line.split("    ")
#     if data[1]=="0":
#         a1 = a1+1
#     if data[1]=="1":
#         a2 = a2+1
#     if data[1]=="2":
#         a3 = a3+1
#     if data[1]=="3":
#         a4 = a4+1
# 
# print a1
# print a2
# print a3
# print a4
print "----------"
for i in range(7):
    print i
    
c = [1,2,3,4,5]
print c[-1]
print c.pop()
print c

print "********"
data = [1.0,2.0,3.0]
newdata = map(float,data)
print newdata
print type(newdata)
# print ",".join(data)

A = [1,2,3,4,5,6]
print A
# print A.indexof(3)
A.pop()
print A

# shuffle()使用样例
import random

x = [i for i in range(10)]
print(x)
random.shuffle(x)
print(x)

print random.randint(0,5)

print "*****************"
A= [5,6,7]
B = [8,9,0]
x = [A,B]
y = copy.copy(x)
A[0] = 11
print x
print y



print "%%%%%%%%%"
# v = [1,1,2,2,3,3,4]
# v1 = [["y"],["s"]]
# print np.array(v1)

# 这是对应这的HMM_2中的数据输入格式的
a1 = np.array([[1],[1],[2],[2],[3]])
a2 = np.array([[2],[3],[4],[5]])
X = []
X.append(a1)
X.append(a2)

print X
print X[0]

b1 = [0,1,1,1,2,3]
b2 = [2,3,4,5]
bb1 = np.array(b1)
bb2 = np.array(b2)
Y = []
Y.append(bb1)
Y.append(bb2)
print Y
print Y[0]
