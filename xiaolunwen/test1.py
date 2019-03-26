# -*- coding: utf-8 -*-

import numpy as np

def t1(Ann,Bnm,Pi,O):
    A = np.array(Ann, np.float)
    B = np.array(Bnm, np.float)
    Pi = np.array(Pi, np.float)
    O = np.array(O, np.float)
    print A
    print B
    print Pi
    print O

if __name__=="__main__":
    N = 3
    M = 2
    Pi = [1 / float(N)] * N
    A = [[1 / float(N) for j in range(N)] for i in range(N)]
    B = [[1 / float(M) for j in range(M)] for i in range(N)]
    t1(A,B,Pi,[0,1,0])
    delta = np.zeros((5, 4), np.float)
    print delta
    a = range(0,2)
    print a
    V = [k for k in range(4)]
    print V