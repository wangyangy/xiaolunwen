#coding=utf8

def readFile(filename):
    fp = open(filename,'r')
    fp1 = open("newData//HTRU_2_.txt",'w')
    for line in fp.readlines():
        line = line+"\n"
        fp1.write(line)
    fp.close()
    fp1.close()

if __name__=="__main__":
    readFile("newData//HTRU_2.txt")
    