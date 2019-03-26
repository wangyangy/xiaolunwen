# -*- coding: utf-8 -*-
import random
    
# NL    账号T时间内登录次数
# NIP    账号T时间内IP地址数
# NC    IP地址所属地区
# NUA    时间T内UserAgent数
# NSL    标记安全登录次数
# NSIP    标记安全的IP数
# RL    标记风险的登录数
# RIP    标记风险的IP数


def createNormalData():
    fp = open("data\\userinfo.txt",'r')
    users = []
    for line in fp.readlines():
        line = line.strip()
        data = line.split("    ")
        username = data[2]
        users.append(username)
    fp.close()
    userData = {}
    for user in users:
        NL = random.randint(14,33)
        NIP = random.randint(3,7)
        NC = random.randint(1,2)
        NUA = random.randint(1,3)
        NSL = random.randint(NL-8,NL)
        NSIP = random.randint(NIP-2,NIP)
        RL = random.randint(0,5)
        if RL==0:
            RIP = 0
        else:
            RIP = random.randint(1,RL)
        data = [NL,NIP,NC,NUA,NSL,NSIP,RL,RIP]
        userData[user] = data
    f = open("classData\\normal.txt",'w')
    for key in userData.keys():
        user = key
        data = userData[user]
        line = user+'    '
        data = map(str,data)
        line += '    '.join(data)
        line+="\n"
        f.write(line)
    f.close()
    
    

def createAbnormalData():
    fp = open("dataproblem\\userinfo.txt",'r')
    users = []
    for line in fp.readlines():
        line = line.strip()
        data = line.split("    ")
        username = data[2]
        users.append(username)
    fp.close()
    userData = {}
    for user in users:
        NL = random.randint(14,33)
        NIP = random.randint(6,10)
        NC = random.randint(2,5)
        NUA = random.randint(1,5)
        NSL = random.randint(NL-10,NL-5)
        NSIP = random.randint(0,3)
        RL = random.randint(0,10)
        if RL==0:
            RIP = 0
        else:
            RIP = random.randint(1,RL)
        data = [NL,NIP,NC,NUA,NSL,NSIP,RL,RIP]
        userData[user] = data
    f = open("classData\\abnormal.txt",'w')
    for key in userData.keys():
        user = key
        data = userData[user]
        line = user+'    '
        data = map(str,data)
        line += '    '.join(data)
        line+="\n"
        f.write(line)
    f.close()
    
def combine():
    fp = open("classData\\normal.txt",'r')
    data = []
    for line in fp.readlines():
        line = line.strip()
        line += "    "+"normal"+"\n"
        data.append(line)
        print line
    fp = open("classData\\abnormal.txt",'r')
    for line in fp.readlines():
        line = line.strip()
        line += "    "+"abnormal"+"\n"
        data.append(line)
    f = open("classData\\combine.csv",'w')
    f.write("username,NL,NIP,NC,NUA,NSL,NSIP,RL,RIP,class"+"\n")
    for d in data:
        d = d.replace("    ",",")
        f.write(d)
    f.close()
    
def combine_normalize():
    fp = open("classData\\normal.txt",'r')
    data = []
    for line in fp.readlines():
        line = line.strip()
        line += "    "+"normal"+"\n"
        data.append(line)
    fp = open("classData\\abnormal.txt",'r')
    for line in fp.readlines():
        line = line.strip()
        line += "    "+"abnormal"+"\n"
        data.append(line)
    #获取某一列的最大值和最小值，用于之后的数据标准化
    max_ = [0,0,0,0,0,0,0,0]
    min_ = [10000,10000,10000,10000,10000,10000,10000,10000]
    for d in data:
        num = d.split("    ")
        for i in range(len(num)):
            if i==0 or i==9:
                continue
            if max_[i-1]<float(num[i]):
                max_[i-1]=float(num[i])
            if min_[i-1]>float(num[i]):
                min_[i-1]=float(num[i])
            
    f = open("classData\\normalize.txt",'w')
    for d in data:
        num = d.split("    ")
        for i in range(len(num)):
            if i==0 or i==9:
                continue
            num[i] = (float)(float(num[i])-min_[i-1])/(float)(max_[i-1]-min_[i-1])
        num = map(str,num)
        num = '    '.join(num)
        f.write(num)
    f.close()
        
        
def julei_fenlei():
    fp = open("classData\\combine.txt",'r')
    user_class = []
    for line in fp.readlines():
        line = line.strip()
        data = line.split("    ")
        user = data[0]
        class_ = data[9]
        item = [user,class_]
        user_class.append(item)
    fp.close()
    fp = open("classData\\kmeansk_3.txt",'r')
    user_cat = []
    for line in fp.readlines():
        line = line.strip()
        data = line.split("    ")
        user = data[0]
        cat = data[1]
        item = [user,cat]
        user_cat.append(item)
    fp.close()
    fp = open("classData\\combine_kmeans.txt",'w')
    for i in user_class:
        for j in user_cat:
            if i[0]==j[0]:
                fp.write(i[0]+"    "+i[1]+"    "+j[1]+"\n")
    fp.close()

def chouquInstance_2():
    fp = open("classData\\kmeansk_2.txt",'r')
    user_cat = {}
    for line in fp.readlines():
        line = line.strip()
        data = line.split("    ")
        user = data[0]
        cat = data[1]
        user_cat[user] = cat
    fp.close()
    fp = open("classData\\combine.txt",'r')
    fp1 = open("classData\\combine_k_2_1.csv",'w')
    fp2 = open("classData\\combine_k_2_2.csv",'w')
    fp1.write("username,NL,NIP,NC,NUA,NSL,NSIP,RL,RIP,class"+"\n")
    fp2.write("username,NL,NIP,NC,NUA,NSL,NSIP,RL,RIP,class"+"\n")
    for line in fp.readlines():
        line = line.strip().split("    ")
        line = ','.join(line)
        data = line.split(",")
        user = data[0]
        if user_cat[user]=="0":
            fp1.write(line+"\n")
        elif user_cat[user]=="1":
            fp2.write(line+"\n")
    fp.close()
    fp1.close()
    fp2.close()



def chouquInstance_3():
    fp = open("classData\\kmeansk_3.txt",'r')
    user_cat = {}
    for line in fp.readlines():
        line = line.strip()
        data = line.split("    ")
        user = data[0]
        cat = data[1]
        user_cat[user] = cat
    fp.close()
    fp = open("classData\\combine.txt",'r')
    fp1 = open("classData\\combine_k_3_1.csv",'w')
    fp2 = open("classData\\combine_k_3_2.csv",'w')
    fp3 = open("classData\\combine_k_3_3.csv",'w')
    fp1.write("username,NL,NIP,NC,NUA,NSL,NSIP,RL,RIP,class"+"\n")
    fp2.write("username,NL,NIP,NC,NUA,NSL,NSIP,RL,RIP,class"+"\n")
    fp3.write("username,NL,NIP,NC,NUA,NSL,NSIP,RL,RIP,class"+"\n")
    for line in fp.readlines():
        line = line.strip().split("    ")
        line = ','.join(line)
        data = line.split(",")
        user = data[0]
        if user_cat[user]=="0":
            fp1.write(line+"\n")
        elif user_cat[user]=="1":
            fp2.write(line+"\n")
        elif user_cat[user]=="2":
            fp3.write(line+"\n")
    fp.close()
    fp1.close()
    fp2.close()
    fp3.close()
    
            
        
if __name__=="__main__":
#     createNormalData()
#     createAbnormalData()
    combine()
#     combine_normalize()
#     chouquInstance_2()
#     chouquInstance_3()
    
            