# -*- coding: utf-8 -*-
import time
import MySQLdb
import random
import operator
import jsonlines
from fake_useragent import UserAgent
import kmeans


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

'''数据库连接函数'''
def getMysqlConn():
    return MySQLdb.connect(host= "10.245.146.90",user="ROOT",passwd= "HITnsLAB203",db="wiki1",port=3306,charset="utf8")

'''获取数据库的用户信息'''
def getData():
    fp = open("dataproblem\\userinfo.txt",'w')
    conn = getMysqlConn()
    cur = conn.cursor()
    sql = "select * from userinfo"
    cur.execute(sql)
    results = cur.fetchall()
    for row in results:
        s=""
        id = str(row[0])
        realname = str(row[1])
        username = str(row[2])
        password = str(row[3])
        uuid = str(row[4])
        createtime = str(row[5])
        salt = str(row[6])
        info = str(row[7])
        userqq = str(row[8])
        telephone = str(row[9])
        number = str(row[10])
        introduction = str(row[11])
        userlevel = str(row[12])
        state = str(row[13])
        s+=id+"    "+realname+"    "+username+"    "+password+"    "+uuid+"    "
        s+=createtime+"    "+salt+"    "+info+"    "+userqq+"    "+telephone+"    "
        s+=number+"    "+introduction+"    "+userlevel+"    "+state+"\n"
        fp.write(s)
    fp.close()
    cur.close()
    conn.close()

'''创建随机时间'''
def createTime():
    a1=(2018,5,16,8,0,0,0,0,0)              #设置开始日期时间元组（1976-01-01 00：00：00）
    a2=(2018,11,00,23,59,59,0,0,0)    #设置结束日期时间元组（1990-12-31 23：59：59）
    
    start=time.mktime(a1)    #生成开始时间戳
    end=time.mktime(a2)      #生成结束时间戳
    
    while True:
        t=random.randint(start,end)    #在开始和结束时间戳中随机取出一个
        date_touple=time.localtime(t)          #将时间戳生成时间元组
        date=time.strftime("%Y-%m-%d %H:%M:%S",date_touple)  #将时间元组转成格式化字符串（1976-05-21）
        date = str(date)
        hour_minutes_second = date.split(" ")[1]
        hour = hour_minutes_second.split(":")[0]
        month = date.split(" ")[0].split("-")[1]
        #去掉凌晨时间
        if hour=="00" or hour=="01" or hour=="02" or hour=="03" or hour=="04" or hour=="05" or hour=="06" or hour=="23":
            continue
        #去掉假期时间
        if month=="08" or month=="02":
            continue
        return date


clientIp = [
            '221.2.164.8','221.2.164.26','221.2.164.47','221.2.164.89',
            '221.2.164.78','221.2.164.66','221.2.164.53','221.2.164.82',
            ]

blacklistIP = ['218.59.162.40' ,'218.59.162.40',
               '211.97.222.210'  ,  '211.97.222.210',
               '61.147.150.170'   , '61.147.150.170',
               '219.146.204.118'  , '219.146.204.118']


'''生成用户登录日志的函数'''   
def createLoginData():
    fp = open("dataproblem\\Logindata_.txt",'w')
    #"session":{"realname":"王阳","state":2,"username":"wangyang@wis-eye.com","userlevel":2},
    fp1 = open("dataproblem\\userinfo.txt",'r')
    realname = []
    username = []
    password = []
    for line in fp1.readlines():
        line = line.strip()
        data = line.split("    ")
        realname.append(data[1])
        username.append(data[2])
        password.append(data[3])
    fp1.close()
    print str(len(username))
    s=""
    for i in range(2283):
        s="{"
        k = random.randint(0,16)
        if k<5:
            index = random.randint(0,16)
        elif k<4:
            index = random.randint(0,2)
        elif k<7:
            index = random.randint(5,5)
        elif k<9:
            index = random.randint(7,8)
        elif k<11:
            index = random.randint(10,11)
        elif k<14:
            index = random.randint(13,13)
        elif k<16:
            index = random.randint(15,16)
            
        s+="\""+"username"+"\""+":"
        s+="\""+str(username[index])+"\","
        s+="\""+"password"+"\""+":"
        s+="\""+str(password[index])+"\","
        s+="\""+"date"+"\""+":"
        s+="\""+createTime()+"\","
        s+="\""+"message"+"\":"
        k = random.randint(0,55)
        if k<8:
            s+="\""+"用户名格式不合法"+"\"}"
        elif k<16:
            s+="\""+"密码错误"+"\"}"
        else:
            s+="\""+"登录成功"+"\"}"
        s+="\n"
        fp.write(s)
    fp.close()

'''产生随机的sessionid'''       
def getRandomStr():
    s = ['1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    t=""
    for i in range(26): 
        index = random.randint(0,35)
        t+=str(s[index])
    return t

'''读取登录日志，姓名，日期，消息，为静态浏览器信息做准备'''
def getUsernameDateMessage():
    fp = open("dataproblem\\Logindata_.txt",'r')
    usernames=[]
    dates=[]
    messages = []
    for line in fp.readlines():
        line = line.replace("{","")
        line = line.replace("}","")
        line = line.replace("\"","")
        data = line.split(",")
        username = data[0].split(":")[1]
        message = data[3].split(":")[1]
        date = data[2].replace("date:","")
        usernames.append(username)
        dates.append(date)
        messages.append(message)
    return usernames,dates,messages

'''随机生成useragent的工具函数'''
def getBrowserInfo():
    ua = UserAgent()
    for i in range(200):
        if i/40==0:
            print ua.ie
        elif i/80==0:
            print ua.firefox
        elif i/120==0:
            print ua.chrome
        elif i/160==0:
            print ua.safari
        else:
            print ua.Opera

'''读取静态userAgent,产生IE,Firefox,Chrom,Safari,Opera等userAgent'''
def getStaticBrowserInfo():
    fp = open("dataproblem\\browser.txt",'r')
    IE = []
    Firefox = []
    Chrom = []
    Safari = []
    Opera = []
    for line in fp.readlines():
        data = line.strip().split(":")
        if "IE" in data[0]:
            IE.append(data[1].strip())
        elif "Firefox" in data[0]:
#             print data[1]
            Firefox.append(data[1])
        elif "Chrome" in data[0]:
#             print data[1]
            Chrom.append(data[1])
        elif "Safari" in data[0]:
            Safari.append(data[1])
        elif "Opera" in data[0]:
            Opera.append(data[1])
    return IE,Firefox,Chrom,Safari,Opera

'''随机生成浏览器插件的工具函数'''      
def getStaticPlugins():
    fp = open("dataproblem\\plugins_.txt",'r')
    plugins = []
    for line in fp.readlines():
        line = line.strip().replace(";","")
        if line not in plugins:
            plugins.append(line)
    fp.close()
    return plugins

'''根据插件信息,数量,用户名为单个用户生成插件列表'''
user_subplugins={}
def getNumberPlugins(plugins,num,user):
    global user_subplugins
#     d.has_key('name')
    if user_subplugins.has_key(user) != True:
        subplugins = random.sample(plugins, num)
        user_subplugins[user] = subplugins
    else:
        #先判断是否要改变
        if random.randint(0,10)<1:
            if len(user_subplugins[user])>20:
                user_subplugins[user].pop()
            elif len(user_subplugins[user])<4:
                while True:
                    p = random.sample(plugins, 1)
                    if p in user_subplugins[user]:
                        continue
                    else:
                        user_subplugins[user].extend(p)
                        break

    return user_subplugins[user]


'''获取所有用户'''
def getAllUser(usernames):
    users = []
    for i in usernames:
        if i not in users:
            users.append(i)
    return users


'''获取所有用户的浏览器userAgent'''
def user_browser(users,IE,Firefox,Chrome,Safari,Opera):
    borwsers = []
    borwsers.extend(IE)
    borwsers.extend(Firefox)
#     print Chrome
    borwsers.extend(Chrome)
    borwsers.extend(Safari)
    borwsers.extend(Opera)
    user_browsers = {}
    for user in users:
        num = random.randint(2,3)
        subBorwsers = random.sample(borwsers, num)
        user_browsers[user] = subBorwsers
#     for eachKey in user_browsers.keys():
#         print eachKey+':', user_browsers[eachKey]
    return user_browsers
         
'''检测函数'''
def check():
    fp1 = open("dataproblem\\BrowserData_.txt",'r')
    username="zhangruiqi@wis-eye.com"
    i=0
    tongji = {}
    for line in fp1.readlines():
        line = line.replace("{","")
        line = line.replace("}","")
        line = line.replace("\"","")
        data = line.split(",")
        name = data[0].split(":")[1] 
        if tongji.has_key(name):
            tongji[name] = tongji[name]+1
        else:
            tongji[name] = 1
#         if username==name:
#             i=i+1
#             print i
#             print line.strip()
#             print data[4]+"----"+data[5]
#             print data[10]
    for key in tongji.keys():
        print key+":"+str(tongji[key])

  
'''生成局域网IP信息'''      
def createIP():
    fp = open("dataproblem\\IP.txt",'w')
    for i in range(1000):
        s = "10.246."
        num1 = random.randint(2,254)
        num2 = random.randint(2,254)
        s+=str(num1)+"."
        s+=str(num2)+"\n"
        fp.write(s)
    fp.close()

'''获取局域网IP'''    
def getIP():
    fp = open("dataproblem\\IP.txt",'r')
    IPS = []
    for line in fp.readlines():
        line = line.strip()
        IPS.append(line)
    fp.close()
    return IPS
  

user_IP = {}
def getUser_IP(user):
    global user_IP
    if user_IP.has_key(user) != True:
        num = random.randint(2,6)
        user_IP[user] = random.sample(clientIp, num)
    return user_IP[user][random.randint(0,len(user_IP[user]))-1]
        
  
'''根据单个用户获取屏幕分辨率'''  
user_wh={}
def getWH(user):
    global user_wh
    width = [800,1600,1024,1280,1440]
    height = [600,900,768,1024,900]
    if user_wh.has_key(user) != True:
        index = random.randint(0,4)
        user_wh[user] = [width[index],height[index]]                
    return user_wh[user]

'''生成浏览器静态信息'''
def createBrowserData():
    IPS = getIP()
    usernames,dates,messages = getUsernameDateMessage()
    users = getAllUser(usernames)
    IE,Firefox,Chrome,Safari,Opera=getStaticBrowserInfo()
    user_browsers = user_browser(users,IE,Firefox,Chrome,Safari,Opera)   
    plugins = getStaticPlugins()
#     print "plugins个数："+str(len(plugins))
    fp1 = open("dataproblem\\BrowserData_.txt",'w')
    for i in range(len(usernames)):
#         print usernames[i]
        #登录失败的布记录数据
        if messages[i].strip()!="登录成功":
#             print messages[i].strip()
            continue
        s="{"
        s+="\""+"username"+"\""+":"
        s+="\""+str(usernames[i])+"\","
        s+="\""+"date"+"\""+":"
        s+="\""+dates[i]+"\","
        s+="\""+"PHPSESSID"+"\""+":"
        s+="\""+str(getRandomStr())+"\","
        s+="\""+"tjRefer"+"\""+":"
        s+="\""+"http:\/\/10.245.146.90\/login.html"+"\","
        s+="\""+"Browser type"+"\""+":"
        subBrowsers = user_browsers[usernames[i]]
        browser = random.sample(subBrowsers, 1)
        if "MSIE" in browser[0]:
            s+="\""+"IE浏览器"+"\"," 
        elif "Firefox" in browser[0]:
            s+="\""+"Firefox浏览器"+"\"," 
        elif "Chrome" in browser[0]:
            s+="\""+"Chrome浏览器"+"\"," 
        elif "Safari" in browser[0]:
            s+="\""+"Safari浏览器"+"\"," 
        elif "Opera" in browser[0]:
            s+="\""+"Opera浏览器"+"\","
        s+="\""+"Browser information"+"\""+":"
        s+="\""+browser[0]+"\","  
        s+="\""+"cookie"+"\""+":"
        s+="\""+"true"+"\","   
        s+="\""+"CPU"+"\""+":"
        s+="\""+"undefined"+"\","
        s+="\""+"Browser MIME"+"\""+":"
        s+="\""+str(random.randint(5,50))+"\","
        s+="\""+"plugins"+"\""+":"
        number = random.randint(5,15)
        subplugins = getNumberPlugins(plugins,number,usernames[i])
        sub = ""
        for item in subplugins:
            sub += str(item)+";"
        s+="\""+sub+"\","
        s+="\""+"plugins number"+"\""+":"
        s+="\""+str(len(subplugins))+"\","
        s+="\""+"language"+"\""+":"
        s+="\""+"undefined"+"\","
        s+="\""+"Screen height"+"\""+":"
        wh = getWH(usernames[i])
        s+="\""+str(wh[1])+"\","
        s+="\""+"Screen width"+"\""+":"
        s+="\""+str(wh[0])+"\","
        s+="\""+"clientIP"+"\""+":"
#         index1 = random.randint(0,len(clientIp)-1)
#         s+="\""+str(clientIp[index1])+"\","
        ip = getUser_IP(usernames[i])
#         print ip
        s+="\""+str(ip)+"\","
        s+="\""+"clientId"+"\""+":"
        s+="\""+str(random.randint(0,1000000))+"\","
        s+="\""+"clientName"+"\""+":"
        s+="\""+"山东省威海市"+"\","
        s+="\""+"url"+"\""+":"
        s+="\""+"http:\/\/10.245.146.90:81\/personal\/indexb.php"+"\","
        s+="\""+"title"+"\""+":"
        s+="\""+"NIT 最懂你的知识共享平台-WIKI"+"\","
        s+="\""+"domain"+"\""+":"
        s+="\""+"10.245.146.90"+"\","
        s+="\""+"IP"+"\""+":"
        index2 = random.randint(0,999)
        s+="\""+str(IPS[index2])+"\","
#         s+="\""+"totalTime"+"\","
#         s+="\""+"[{}]"+"\""
        s+="\""+"class"+"\""+":"
        s+="\""+str(1)+"\""
        s+="}"+"\n"
        fp1.write(s)
    fp1.close()
        
'''生成网页动态浏览信息'''

def createtime1(index):
    time1 = random.randint(0, 600)
    if index < 12:
        time1 = random.randint(300, 400)
    elif index < 20:
        time1 = random.randint(100, 200)
    elif index < 40:
        time1 = random.randint(150, 300)
    elif index < 64:
        time1 = random.randint(450, 580)
    if random.randint(0, 50) < 20:
        time1 = 0
    return time1


def createtime2(index):
    time2 = random.randint(1000, 3600)
    if index < 12:
        time2 = random.randint(1500, 1900)
    elif index < 20:
        time2 = random.randint(2000, 2800)
    elif index < 40:
        time2 = random.randint(1000, 1800)
    elif index < 64:
        time2 = random.randint(2500, 3600)
    if random.randint(0, 50) < 20:
        time2 = 0
    return time2


def createtime3(index):
    time3 = random.randint(0, 100)
    if index < 12:
        time3 = random.randint(27, 48)
    elif index < 20:
        time3 = random.randint(20, 31)
    elif index < 40:
        time3 = random.randint(50, 78)
    elif index < 64:
        time3 = random.randint(60, 100)
    if random.randint(0, 50) < 30:
        time3 = 0
    return time3


def createtime4(index):
    time4 = random.randint(30, 300)
    if index < 12:
        time4 = random.randint(120, 188)
    elif index < 20:
        time4 = random.randint(60, 131)
    elif index < 32:
        time4 = random.randint(20, 78)
    elif index < 40:
        time4 = random.randint(48, 108)
    elif index < 52:
        time4 = random.randint(88, 168)
    elif index < 64:
        time4 = random.randint(150, 279)
    return time4


def createClick(index):
    if index < 8:
        click = random.randint(5, 13)
    elif index < 20:
        click = random.randint(15, 25)
    elif index < 40:
        click = random.randint(9, 19)
    elif index < 64:
        click = random.randint(16, 40)
    if random.randint(0, 21) < 3:
        click = 0
    return click


def createKeyUp(time2):
    if time2 == 0:
        keyup = random.randint(0, 70)
    elif time2 < 500:
        keyup = random.randint(100, 700)
    elif time2 < 1000:
        keyup = random.randint(500, 1600)
    elif time2 < 1500:
        keyup = random.randint(800, 2100)
    elif time2 < 2000:
        keyup = random.randint(1200, 2700)
    else:
        keyup = random.randint(1900, 4600)
    return keyup


def createWheel(index):
    if index < 12:
        wheel = random.randint(0, 90)
    elif index < 20:
        wheel = random.randint(0, 180)
    elif index < 40:
        wheel = random.randint(0, 300)
    elif index < 64:
        wheel = random.randint(50, 400)
    if random.randint(0, 12) < 5:
        wheel = 0
    return wheel

'''创建动态数据'''
def createDynamicData():
    usernames,dates,messages = getUsernameDateMessage()
    users = getAllUser(usernames)
    fp = open("dataproblem\\dynamic.txt",'w')
    fp1 = open("dataproblem\\BrowserData_.txt",'r')
    fp.write("username    date    PHPSESSID    time1(文档浏览时间)    time2(文档编辑时间)    time3(任务，资料编辑时间)    time4(其余时间)    click(所有页面的总和)    keyup(所有页面的总和)    wheel(所有页面的总和,滚动距离)"+"\n")
#     {"username":"liujiahow@wis-eye.com","date":"2017-11-10 10:24:46","PHPSESSID":"6h6jd74att4o32xj3v1787gqc6","tjRefer":"http:\/\/10.245.146.90\/login.html","浏览器类型":"Chrome浏览器","浏览器属性信息":"Mozilla/5.0 (X11; CrOS i686 4319.74.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.57 Safari/537.36","浏览器是否启用了cookie","true","CPU等级","undefined","浏览器的MIME类型","46","安装的插件数目","6","插件的名称","NVIDIA 3D VISION;Alipay security control;iTrusChina iTrusPTA,XEnroll,iEnroll,hwPTA,UKeyInstalls Firefox PluginMicrosoft Office 2013;Chromium PDF Plugin;Infinity新标签页;Chrome PDF Plugin;","语言","undefined","屏幕分辨率高度","600","屏幕分辨率宽度","800","clientIP","221.2.164.7","clientId","499989","clientName","山东省威海市","url","http:\/\/10.245.146.90:81\/personal\/indexb.php","title","NIT 最懂你的知识共享平台-WIKI","domain","10.245.146.90","IP","10.246.71.177",}
    #内部文档浏览，文档编辑管理，任务领取管理，其他（成员查看，个人信息管理等等）
    #增加键盘输入统计，鼠标左键点击统计，滑轮统计（滑轮由上到下大约数值为30）
    for line in fp1.readlines():
        s=""
        line = line.replace("{","").replace("}","").replace("\"","")
        data = line.split(",")
        username = data[0].split(":")[1]
        date = data[1].replace("date:","")
        PHPSESSID = data[2].split(":")[1]
        index = users.index(username)
        time1 = createtime1(index)
        time2 = createtime2(index)
        time3 = createtime3(index)
        time4 = createtime4(index)
        if time1+time2+time3==0:
            #不能全为0（概率很小）
            time1 = createtime1(index)
            time2 = createtime2(index)
            time3 = createtime3(index)
        s+=username+"    "
        s+=date+"    "
        s+=PHPSESSID+"    "
        s+=str(time1)+"    "
        s+=str(time2)+"    "
        s+=str(time3)+"    "
        s+=str(time4)+"    "
        click = createClick(index)
        s+=str(click)+"    "
        keyup = createKeyUp(time2)
        s+=str(keyup)+"    "
        wheel = createWheel(index)
        s+=str(wheel)+"    "
        s+="\n"
        fp.write(s)
    fp.close()
        
          

'''根据时间进行排序登录信息'''
def paixuLogindata_time():
    fp = open("dataproblem\\Logindata_.txt",'r')
    fp1 = open("dataproblem\\Logindata_paixu.txt",'w')
    lines = {}
    for line in fp.readlines():
        linecopy = line
        line = line.replace("{","").replace("}","").replace("\"","")
        data = line.split(",")
        date = data[2].replace("date:","")
        t = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(t))
#         print timeStamp
        lines[timeStamp] = linecopy
    fp.close()       
    res = sorted(lines.items(),key=lambda p:p[0],reverse=False)
    for item in res:
#         print item
        fp1.write(item[1])
    fp1.close()

'''根据时间进行排序浏览器信息'''
def paixuBrowserdata_time():
    fp = open("dataproblem\\BrowserData_.txt",'r')
    fp1 = open("dataproblem\\BrowserData_paixu.txt",'w')
    lines = {}
    for line in fp.readlines():
        linecopy = line
        line = line.replace("{","").replace("}","").replace("\"","")
        data = line.split(",")
        date = data[1].replace("date:","")
        t = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(t))
#         print timeStamp
        lines[timeStamp] = linecopy
    fp.close()       
    res = sorted(lines.items(),key=lambda p:p[0],reverse=False)
    for item in res:
#         print item
        fp1.write(item[1])
    fp1.close()    
    
  
'''将所有静态数据整合到一起，威胁情报数据也整合'''
def combine():
    pass   

'''获取登录错误信息'''
def getLoginError():
    usernames,dates,messages = getUsernameDateMessage()
    username_error = {}
    for i in range(len(messages)):
        if "密码错误"==messages[i].strip() or "用户名格式不合法"==messages[i].strip():
            if username_error.has_key(usernames[i]):
                username_error[usernames[i]]=username_error[usernames[i]]+1
            else:
                username_error[usernames[i]]=1
    k=0
    for key in username_error.keys():
       k+=username_error[key] 
#     print str(k)
    return username_error


'''数据预处理，根据人头将数据平均化，之后在进行标准化'''
def prepareAndnormalize():
    datas={}
    fp = open("dataproblem\\dynamic.txt",'r')
    for line in fp.readlines():
        if "username" in line:
            continue
        data = line.strip().split("    ")
        username = data[0]
        time1 = int(data[3])
        time2 = int(data[4])
        time3 = int(data[5])
        time4 = int(data[6])
        click = int(data[7])
        keyup = int(data[8])
        wheel = int(data[9])
        data = [time1,time2,time3,time4,click,keyup,wheel]
        if datas.has_key(username)==True:   
            datas[username].append(data)
        else:
            datas[username]=[]
            datas[username].append(data)
    max_ = [0,0,0,0,0,0,0]
    min_ = [10000,10000,10000,10000,10000,10000,10000]
    average = {}
    for key in datas.keys():
        username = key
        datas_username = datas[key]
        sum = [0,0,0,0,0,0,0]
        subaverage = [0,0,0,0,0,0,0]
        for i in datas_username:
            for j in range(7):
                sum[j]+=i[j]
                if max_[j]<i[j]:
                    max_[j]=i[j]
                if min_[j]>i[j]:
                    min_[j]=i[j]
        
        for j in range(7):
            subaverage[j] = sum[j]/len(datas_username)
#         print subaverage
        average[username] = subaverage
#     print average
    normalize = {}
    for key in average.keys():
        junzhi = average[key]
        a = [0,0,0,0,0,0,0]
        for i in range(7):
            a[i] = (float)(junzhi[i]-min_[i])/(float)(max_[i]-min_[i])
        normalize[key] = a
        print normalize[key]
#     print normalize
#     print len(normalize)
    return normalize

'''获取标准化的用户动态数据'''
def getnormalizedata():
    normalizedata = prepareAndnormalize()
    fp = open("dataproblem\\normalizedata.txt",'w')
    for key in normalizedata.keys():
        value = normalizedata[key]
        value = map(str,value)
        fp.write(key)
        fp.write("    ")
        fp.write(str('    '.join(value)))
        fp.write("\n")
    fp.close()
    fp = open("dataproblem\\normalizedata.txt",'r')
    userdata = {}
    for line in fp.readlines():
        data=line.strip().split("    ")
        username = data[0]
        userdata[username] = data[1:len(data)]
    return userdata

'''解析json数据，获取个人信息列表'''
def parseJson():
    username_error = getLoginError()
    userDatas = []
    broswers = {}
    user_Agents = {}
    plugins_ = {}
    wh = {}
    IPS = {}
    locations = {}
    with open("dataproblem\\BrowserData_.txt", "r") as f:
        for item in jsonlines.Reader(f):
            username = item['username']
            broswer = item['Browser type']
            user_Agent = item['Browser information']
            plugins = item['plugins']
            h = item['Screen height']
            w = item['Screen width']
            IP = item['clientIP']
            location = item['clientName']
            if broswers.has_key(username):
                if broswer not in broswers[username]:
                    broswers[username].append(broswer)
            else:
                broswers[username] = [broswer]
            if user_Agents.has_key(username):
                if user_Agent not in user_Agents[username]:
                    user_Agents[username].append(user_Agent)
            else:
                user_Agents[username] = [user_Agent]
            if plugins_.has_key(username):
                if plugins not in plugins_[username]:
                    plugins_[username].append(plugins)
            else:
                plugins_[username] = [plugins]
            if wh.has_key(username):
                if (w,h) not in wh[username]:
                    wh[username].append((w,h))
            else:
                wh[username] = [(w,h)]
            if IPS.has_key(username):
                if IP not in IPS[username]:
                    IPS[username].append(IP)
            else:
                IPS[username] = [IP]
            if locations.has_key(username):
                if location not in locations[username]:
                    locations[username].append(location)
            else:
                locations[username] = [location]
                
    usernames,dates,messages = getUsernameDateMessage()
    users = getAllUser(usernames)
    user_dynamicdata = getnormalizedata()
    fp1 = open("dataproblem\\userDataTongji.txt",'w')    
    for user in users:
        browser = broswers[user]
        user_Agent = user_Agents[user]
        plugins = plugins_[user]
        wh_ = wh[user]
        ip = IPS[user]
        local = locations[user]
        if username_error.has_key(user):
            message = username_error[user]
        else:
            message = 0
        dynamicdata = user_dynamicdata[user]
        fp1.write(user+":\n")
        fp1.write("    "+"browser"+"    "+str('    '.join(browser))+"\n")
        fp1.write("    "+"userAgent"+"    "+str('    '.join(user_Agent))+"\n")
        fp1.write("    "+"plugin"+"    "+str('    '.join(plugins))+"\n")
        fp1.write("    "+"Screen"+"    "+str(wh_[0][0])+"*"+str(wh_[0][1])+"\n")
        fp1.write("    "+"ip"+"    "+str('    '.join(ip))+"\n")
        fp1.write("    "+"local"+"    "+str('    '.join(local))+"\n")
        fp1.write("    "+"login abnormal"+"    "+str(message)+"\n")
        fp1.write("    "+"dynamic data"+"    "+str('    '.join(dynamicdata))+"\n")
        
    return userDatas        


'''数据预处理，每条数据都进行标准化'''
def getMax_Min():
    datas=[]
    fp = open("data\\dynamic.txt",'r')
    for line in fp.readlines():
        if "username" in line:
            continue
        data = line.strip().split("    ")
        time1 = int(data[3])
        time2 = int(data[4])
        time3 = int(data[5])
        time4 = int(data[6])
        click = int(data[7])
        keyup = int(data[8])
        wheel = int(data[9])
        data = [time1,time2,time3,time4,click,keyup,wheel]
        datas.append(data)
    max_ = [0,0,0,0,0,0,0]
    min_ = [10000,10000,10000,10000,10000,10000,10000]
    for d in datas:
        for j in range(7):
            if max_[j]<d[j]:
                max_[j]=d[j]
            if min_[j]>d[j]:
                min_[j]=d[j]
    return max_,min_ 


def parseJson_combine():
    browserData = {}
    username_error = getLoginError()
    with open("dataproblem\\BrowserData_.txt", "r") as f:
        for item in jsonlines.Reader(f):
            session = item['PHPSESSID']
            username = item['username']
            browser = item['Browser type']
            h = item['Screen height']
            w = item['Screen width']
            IP = item['clientIP']
            browserData[session] = [username,browser,h,w,IP]
    dynamicData = {}
    max_,min_ = getMax_Min()
    fp = open("dataproblem\\dynamic.txt",'r')
    for line in fp.readlines():
        if "username" in line:
            continue
        data = line.strip().split("    ")
        username = data[0]
        session = data[2]
        time1 = int(data[3])
        time2 = int(data[4])
        time3 = int(data[5])
        time4 = int(data[6])
        click = int(data[7])
        keyup = int(data[8])
        wheel = int(data[9])
        data = [session,time1,time2,time3,time4,click,keyup,wheel]
        #第一个是session要去掉
        for i in range(len(data)-1):
            data[i+1] = (float)(data[i+1]-min_[i])/(float)(max_[i]-min_[i])           
        dynamicData[session] = data
    fp = open("combineData\\combineData_Abnormal.txt","w")
    fp.write("username    "+"BrowserType    "+"ScreenHeight    "+"ScreenWidth    "+"clientIP    "+"time1(文档浏览时间)    time2(文档编辑时间)    time3(任务，资料编辑时间)    time4(其余时间)    click(所有页面的总和)    keyup(所有页面的总和)    wheel(所有页面的总和,滚动距离)    loginAbnormal    class"+"\n")
    for key in browserData.keys():
        session = key
        data = browserData[key]
        for key1 in dynamicData.keys():
            if key1==key:
                data1 = dynamicData[key1]
                data.extend(data1)
                data = map(str,data)
                line = '    '.join(data)
                loginAbnormal = username_error[data[0]]
                line += "    "+str(loginAbnormal) 
                line += "    "+str(1)+"\n"
                fp.write(line)



        
if __name__=="__main__":
# #     #步骤一 
#     createLoginData()
#     paixuLogindata_time()
#      #步骤二
#     createBrowserData()
#     check()
#     paixuBrowserdata_time()
#     步骤三
#     createDynamicData()
    #步骤四，生成用户数据列表
    #分类时使用该函数
#     parseJson()
    parseJson_combine()
    
    
            