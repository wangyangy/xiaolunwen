# -*- coding: utf-8 -*-
import time
import MySQLdb
import random
import operator
import json


def readfile(file):
    fp = open(file,'r')
    datas = []
    username = {}
    for line in fp.readlines():
        line = line.strip()
        l = json.dumps(line)
        data = json.loads(l)
        # print data["session"]
        print data
        print type(data)




if __name__=="__main__":
    readfile("newdata2019_3_18//Logindata.txt")
