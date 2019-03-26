# -*- coding:utf-8 -*-
# 隐马尔科夫链模型
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq
import numpy as np
from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv
from abc import ABCMeta, abstractmethod
from sklearn import cluster

class _BaseHMM():
    """
    基本HMM虚类，需要重写关于发射概率的相关虚函数
    N : 隐藏状态的数目
    n_iter : 迭代次数
    M : 观测值维度
    PI : 初始概率
    A : 状态转换概率
    """
    __metaclass__ = ABCMeta  # 虚类声明

    def __init__(self, N=1, M=1, iter=20):
        self.N = N
        self.M = M
        self.PI = np.ones(N) * (1.0 / N)  # 初始状态概率
        # self.PI = np.array([0.15,0.1,0.5,0.25]) # 初始状态概率
        self.A = np.ones((N, N)) * (1.0 / N)  # 状态转换概率矩阵
        # print self.A
        self.trained = False # 是否需要重新训练
        self.n_iter = iter  # EM训练的迭代次数

    # 初始化发射参数
    @abstractmethod
    def _init(self,X):
        pass

    # 虚函数：返回发射概率
    @abstractmethod
    def emit_prob(self, x):  # 求x在状态k下的发射概率 P(X|Z)
        return np.array([0])

    # 虚函数
    @abstractmethod
    def generate_x(self, z): # 根据隐状态生成观测值x p(x|z)
        return np.array([0])

    # 虚函数：发射概率的更新
    @abstractmethod
    def emit_prob_updated(self, X, post_state):
        pass

    # 通过HMM生成序列
    def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.M))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.N, 1, p=self.PI)  # 采样初始状态
        X[0] = self.generate_x(Z_pre) # 采样得到序列第一个值
        Z[0] = Z_pre

        for i in range(seq_length):
            if i == 0: continue
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            Z_next = np.random.choice(self.N, 1, p=self.A[Z_pre,:][0])
            Z_pre = Z_next
            # P(Xn+1|Zn+1)
            X[i] = self.generate_x(Z_pre)
            Z[i] = Z_pre

        return X,Z

    # 估计序列X出现的概率
    def X_prob(self, X, Z_seq=np.array([])):
        # 状态序列预处理
        # 判断是否已知隐藏状态
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.N))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.N))
        # 向前向后传递因子
        _, c = self.forward(X, Z)  # P(x,z)
        # 序列的出现概率估计
        prob_X = np.sum(np.log(c))  # P(X)
        return prob_X

    # 已知当前序列预测未来（下一个）观测值的概率
    def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
        if self.trained == False or istrain == False:  # 需要根据该序列重新训练
            self.train(X)

        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.N))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.N))
        # 向前向后传递因子
        alpha, _ = self.forward(X, Z)  # P(x,z)
        prob_x_next = self.emit_prob(np.array([x_next]))*np.dot(alpha[X_length - 1],self.A)
        return prob_x_next

    def decode(self, X, istrain=True):
        """
        利用维特比算法，已知序列求其隐藏状态值
        :param X: 观测值序列
        :param istrain: 是否根据该序列进行训练
        :return: 隐藏状态序列
        """
        if self.trained == False or istrain == False:  # 需要根据该序列重新训练
            self.train(X)

        X_length = len(X)  # 序列长度
        state = np.zeros(X_length)  # 隐藏状态

        pre_state = np.zeros((X_length, self.N))  # 保存转换到当前隐藏状态的最可能的前一状态
        max_pro_state = np.zeros((X_length, self.N))  # 保存传递到序列某位置当前状态的最大概率

        _,c=self.forward(X,np.ones((X_length, self.N)))
        max_pro_state[0] = self.emit_prob(X[0]) * self.PI * (1/c[0]) # 初始概率

        # 前向过程
        for i in range(X_length):
            if i == 0: continue
            for k in range(self.N):
                prob_state = self.emit_prob(X[i])[k] * self.A[:,k] * max_pro_state[i-1]
                max_pro_state[i][k] = np.max(prob_state)* (1/c[i])
                pre_state[i][k] = np.argmax(prob_state)

        # 后向过程
        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1,:])
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        # state_string = []
        # for i in state:
        #     if i==0:
        #         state_string.append("s1")
        #     elif i==1:
        #         state_string.append("s2")
        #     elif i==2:
        #         state_string.append("s3")
        #     else:
        #         state_string.append("s4")
        return  state
        # return state_string

    # 针对于多个序列的训练问题
    def train_batch(self, X, Z_seq=list()):
        # 针对于多个序列的训练问题，其实最简单的方法是将多个序列合并成一个序列，而唯一需要调整的是初始状态概率
        # 输入X类型：list(array)，数组链表的形式
        # 输入Z类型: list(array)，数组链表的形式，默认为空列表（即未知隐状态情况）
        self.trained = True
        X_num = len(X) # 序列个数
        self._init(self.expand_list(X)) # 发射概率的初始化

        # 状态序列预处理，将单个状态转换为1-to-k的形式
        # 判断是否已知隐藏状态
        if Z_seq==list():
            Z = []  # 初始化状态序列list
            for n in range(X_num):
                Z.append(list(np.ones((len(X[n]), self.N))))
        else:
            Z = []
            for n in range(X_num):
                Z.append(np.zeros((len(X[n]),self.N)))
                for i in range(len(Z[n])):
                    Z[n][i][int(Z_seq[n][i])] = 1

        for e in range(self.n_iter):  # EM步骤迭代
            # 更新初始概率过程
            #  E步骤
            # print "iter: ", e
            b_post_state = []  # 批量累积：状态的后验概率，类型list(array)
            b_post_adj_state = np.zeros((self.N, self.N)) # 批量累积：相邻状态的联合后验概率，数组
            b_PI = np.zeros(self.N) # 批量累积初始概率
            for n in range(X_num): # 对于每个序列的处理
                X_length = len(X[n])
                alpha, c = self.forward(X[n], Z[n])  # P(x,z)
                beta = self.backward(X[n], Z[n], c)  # P(x|z)

                post_state = alpha * beta / np.sum(alpha * beta) # 归一化！
                b_post_state.append(post_state)
                post_adj_state = np.zeros((self.N, self.N))  # 相邻状态的联合后验概率
                for i in range(X_length):
                    if i == 0: continue
                    if c[i]==0: continue
                    post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
                                                            beta[i] * self.emit_prob(X[n][i])) * self.A

                if np.sum(post_adj_state)!=0:
                    post_adj_state = post_adj_state/np.sum(post_adj_state)  # 归一化！
                b_post_adj_state += post_adj_state  # 批量累积：状态的后验概率
                b_PI += b_post_state[n][0] # 批量累积初始概率

            # M步骤，估计参数，最好不要让初始概率都为0出现，这会导致alpha也为0
            b_PI += 0.001*np.ones(self.N)
            self.PI = b_PI / np.sum(b_PI)
            b_post_adj_state += 0.001
            for k in range(self.N):
                if np.sum(b_post_adj_state[k])==0: continue
                self.A[k] = b_post_adj_state[k] / np.sum(b_post_adj_state[k])

            self.emit_prob_updated(self.expand_list(X), self.expand_list(b_post_state))

    def expand_list(self, X):
        # 将list(array)类型的数据展开成array类型
        C = []
        for i in range(len(X)):
            C += list(X[i])
        return np.array(C)

    # 针对于单个长序列的训练
    def train(self, X, Z_seq=np.array([])):
        # 输入X类型：array，数组的形式
        # 输入Z类型: array，一维数组的形式，默认为空列表（即未知隐状态情况）
        self.trained = True
        X_length = len(X)
        self._init(X)

        # 状态序列预处理
        # 判断是否已知隐藏状态
        if Z_seq.any():
            Z = np.zeros((X_length, self.N))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.N))

        for e in range(self.n_iter):  # EM步骤迭代
            # 中间参数
            # print e, " iter"
            # E步骤
            # 向前向后传递因子
            alpha, c = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z, c)  # P(x|z)

            post_state = alpha * beta
            post_adj_state = np.zeros((self.N, self.N))  # 相邻状态的联合后验概率
            for i in range(X_length):
                if i == 0: continue
                if c[i]==0: continue
                post_adj_state += (1 / c[i])*np.outer(alpha[i - 1],beta[i]*self.emit_prob(X[i]))*self.A

            # M步骤，估计参数
            self.PI = post_state[0] / np.sum(post_state[0])
            for k in range(self.N):
                self.A[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.emit_prob_updated(X, post_state)

    # 求向前传递因子
    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.N))  # P(x,z)
        alpha[0] = self.emit_prob(X[0]) * self.PI * Z[0] # 初始值
        # 归一化因子
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        # 递归传递
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.A) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i]==0: continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    # 求向后传递因子
    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.N))  # P(x|z)
        beta[X_length - 1] = np.ones((self.N))
        # 递归传递
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.A.T) * Z[i]
            if c[i+1]==0: continue
            beta[i] = beta[i] / c[i + 1]

        return beta



class DiscreteHMM(_BaseHMM):
    """
    发射概率为离散分布的HMM
    参数：
    B : 离散概率分布
    x_num：表示观测值的种类
    此时观测值大小M默认为1
    """
    def __init__(self, N=1, x_num=1, iter=20):
        _BaseHMM.__init__(self, N=N, M=1, iter=iter)
        self.B = np.ones((N, x_num)) * (1.0/x_num)  # 初始化发射概率均值
        self.x_num = x_num

    def _init(self, X):
        self.B = np.random.random(size=(self.N,self.x_num))
        for k in range(self.N):
            self.B[k] = self.B[k]/np.sum(self.B[k])

    def emit_prob(self, x): # 求x在状态k下的发射概率
        prob = np.zeros(self.N)
        for i in range(self.N):
            # print x[0]
            prob[i]=self.B[i][int(x[0])]
        return prob

    def generate_x(self, z): # 根据状态生成x p(x|z)
        return np.random.choice(self.x_num, 1, p=self.B[z][0])

    def emit_prob_updated(self, X, post_state): # 更新发射概率
        self.B = np.zeros((self.N, self.x_num))
        X_length = len(X)
        for n in range(X_length):
            self.B[:,int(X[n])] += post_state[n]

        self.B+= 0.1/self.x_num
        for k in range(self.N):
            if np.sum(post_state[:,k])==0: continue
            self.B[k] = self.B[k]/np.sum(post_state[:,k])

'''
N:s1(异常，活跃),s2（异常，不活跃）,s3（正常，活跃）,s4（正常，不活跃）
M:0,1,2,3
0: 风险访问，活跃
1: 正常访问，不活跃
2: 风险访问，不活跃
3: 正常访问，活跃
#给出40个例子，30个正常（20个正常活跃，10个正常不活跃），10个异常（6个异常活跃，4个异常不活跃）
#[s1:0.15, s4:0.25, s2:0.1, s3:0.5]
'''
def prepareData():
    data1 = np.array([[3], [2], [0], [1], [3], [1], [3], [3], [3], [1], [2], [3], [3], [1], [3]])
    data2 = np.array([[1], [3], [2], [3], [1], [1], [3], [1], [2], [3], [1], [1], [3], [2], [3]])
    data3 = np.array([[3], [3], [0], [1], [1], [2], [3], [1], [1], [1], [3], [3], [2], [3], [3]])
    data4 = np.array([[2], [3], [3], [3], [1], [1], [3], [3], [3], [1], [3], [1], [3], [0], [3]])
    data5 = np.array([[3], [1], [3], [1], [1], [3], [1], [3], [1], [2], [1], [3], [0], [3], [2]])
    data6 = np.array([[3], [3], [1], [2], [1], [3], [0], [3], [1], [2], [3], [3], [2], [0], [3]])
    data7 = np.array([[1], [3], [1], [3], [2], [3], [3], [1], [2], [2], [3], [0], [1], [3], [3]])
    data8 = np.array([[1], [3], [1], [3], [2], [1], [3], [0], [1], [1], [3], [3], [3], [2], [3]])
    data9 = np.array([[3], [1], [1], [1], [3], [3], [3], [3], [2], [3], [2], [3], [3], [3], [1]])
    data10 = np.array([[3], [3], [1], [2], [3], [2], [3], [1], [3], [1], [3], [1], [2], [3], [2]])
    data11 = np.array([[3], [1], [3], [2], [3], [2], [1], [3], [3], [0], [3], [3], [1], [2], [3]])
    data12 = np.array([[1], [3], [1], [3], [3], [0], [2], [2], [3], [3], [3], [1], [1], [3], [3]])
    data13 = np.array([[3], [1], [3], [3], [0], [1], [3], [3], [1], [3], [3], [0], [2], [3], [0]])
    data14 = np.array([[1], [3], [3], [3], [1], [3], [1], [3], [2], [3], [2], [3], [3], [2], [1]])
    data15 = np.array([[3], [3], [3], [2], [3], [3], [0], [1], [3], [1], [3], [2], [1], [3], [3]])
    data16 = np.array([[3], [1], [1], [2], [3], [3], [2], [3], [3], [2], [3], [2], [3], [1], [3]])
    data17 = np.array([[3], [3], [2], [1], [3], [1], [3], [1], [3], [3], [3], [1], [3], [3], [2]])
    data18 = np.array([[3], [3], [2], [1], [3], [3], [1], [3], [1], [0], [3], [0], [1], [2], [3]])
    data19 = np.array([[3], [1], [3], [0], [3], [1], [1], [3], [3], [2], [3], [3], [2], [3], [0]])
    data20 = np.array([[3], [3], [0], [2], [3], [1], [2], [3], [3], [1], [1], [3], [3], [0], [1]])
    data21 = np.array([[1], [1], [2], [1], [2], [3], [0], [1], [1], [3], [1], [3], [1], [1], [2]])
    data22 = np.array([[1], [3], [1], [3], [1], [1], [0], [3], [1], [1], [0], [1], [0], [3], [1]])
    data23 = np.array([[1], [1], [0], [3], [1], [3], [1], [3], [1], [1], [3], [2], [1], [0], [1]])
    data24 = np.array([[1], [1], [3], [2], [1], [1], [2], [1], [1], [3], [1], [0], [1], [2], [1]])
    data25 = np.array([[3], [1], [1], [0], [1], [2], [1], [1], [2], [1], [3], [2], [1], [1], [1]])
    data26 = np.array([[1], [1], [0], [3], [3], [1], [2], [3], [1], [2], [2], [1], [1], [3], [1]])
    data27 = np.array([[1], [1], [3], [2], [1], [1], [0], [1], [2], [2], [3], [3], [2], [3], [1]])
    data28 = np.array([[1], [3], [1], [1], [3], [1], [1], [0], [2], [1], [1], [2], [3], [3], [1]])
    data29 = np.array([[1], [1], [0], [3], [3], [1], [1], [2], [3], [1], [3], [1], [1], [3], [1]])
    data30 = np.array([[1], [1], [3], [2], [1], [1], [3], [0], [1], [0], [1], [2], [1], [1], [3]])
    data31 = np.array([[0], [1], [0], [1], [1], [0], [0], [3], [1], [0], [1], [2], [0], [1], [0]])
    data32 = np.array([[0], [1], [0], [1], [0], [0], [2], [3], [2], [0], [0], [0], [2], [1], [0]])
    data33 = np.array([[0], [0], [3], [2], [0], [2], [0], [0], [1], [0], [2], [0], [2], [0], [0]])
    data34 = np.array([[0], [2], [0], [2], [0], [0], [3], [1], [0], [0], [1], [3], [0], [0], [3]])
    data35 = np.array([[0], [2], [2], [0], [0], [2], [1], [0], [0], [3], [1], [0], [0], [1], [0]])
    data36 = np.array([[0], [0], [1], [0], [1], [0], [2], [0], [0], [2], [0], [0], [2], [0], [0]])
    data37 = np.array([[2], [2], [1], [0], [2], [2], [1], [2], [2], [1], [1], [2], [2], [3], [0]])
    data38 = np.array([[2], [2], [0], [0], [2], [3], [1], [1], [2], [2], [1], [2], [1], [2], [2]])
    data39 = np.array([[2], [2], [0], [0], [2], [3], [2], [1], [2], [2], [1], [2], [2], [1], [3]])
    data40 = np.array([[2], [1], [0], [1], [2], [2], [3], [0], [2], [1], [2], [3], [2], [0], [1],[3]])
    #这里序列的长度是可以不固定的
    Datas = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,
             data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28,
             data29,data30,data31,data32,data33,data34,data35,data36,data37,data38,data39,data40]
    # print Datas
    return Datas


state_list = {'s1':0,'s2':1,'s3':2,'S4':3}

def prepareData_new():
    data0 = np.array([[2], [0], [0], [1], [5], [4], [4], [1], [2], [1], [3], [0], [0], [0], [0]])
    data1 = np.array([[2], [0], [0], [0], [1], [2], [2], [2], [1], [0], [0], [0], [1], [0], [1]])
    data2 = np.array([[0], [1], [0], [2], [1], [1], [0], [3], [0], [2], [1], [1], [1], [5], [0]])
    data3 = np.array([[1], [1], [0], [2], [2], [1], [0], [1], [1], [2], [2], [0], [2], [2], [5]])
    data4 = np.array([[5], [0], [1], [0], [1], [1], [2], [2], [2], [1], [1], [2], [1], [2], [0]])
    data5 = np.array([[2], [2], [2], [2], [2], [3], [2], [2], [2], [0], [0], [0], [0], [1], [0]])
    data6 = np.array([[0], [0], [0], [2], [4], [0], [2], [2], [0], [2], [2], [5], [1], [1], [1]])
    data7 = np.array([[2], [1], [3], [2], [2], [0], [2], [5], [2], [2], [0], [1], [2], [0], [2]])
    data8 = np.array([[1], [1], [0], [1], [0], [2], [0], [2], [2], [1], [3], [0], [2], [0], [2]])
    data9 = np.array([[0], [5], [0], [0], [1], [2], [2], [2], [0], [2], [0], [0], [0], [0], [0]])
    data10 = np.array([[1], [1], [0], [2], [0], [3], [0], [0], [0], [2], [0], [0], [2], [0], [4]])
    data11 = np.array([[1], [3], [0], [5], [2], [1], [0], [2], [2], [0], [2], [2], [1], [5], [2]])
    data12 = np.array([[0], [3], [2], [1], [2], [1], [0], [2], [2], [2], [3], [3], [1], [0], [1]])
    data13 = np.array([[5], [0], [4], [0], [2], [0], [0], [2], [1], [3], [1], [1], [1], [2], [1]])
    data14 = np.array([[1], [2], [1], [1], [2], [5], [1], [0], [2], [2], [2], [0], [2], [2], [1]])
    data15 = np.array([[0], [1], [0], [1], [2], [2], [0], [1], [2], [1], [0], [0], [0], [2], [5]])
    data16 = np.array([[0], [2], [4], [0], [1], [2], [0], [2], [1], [0], [0], [1], [1], [1], [0]])
    data17 = np.array([[1], [1], [1], [0], [2], [2], [1], [0], [1], [0], [0], [1], [0], [4], [1]])
    data18 = np.array([[2], [0], [1], [1], [0], [1], [1], [0], [0], [1], [0], [1], [3], [1], [2]])
    data19 = np.array([[1], [2], [0], [0], [0], [2], [1], [2], [0], [2], [2], [2], [0], [2], [0]])
    data20 = np.array([[2], [1], [2], [0], [1], [1], [2], [0], [0], [0], [2], [5], [2], [2], [1]])
    data21 = np.array([[2], [0], [2], [2], [1], [1], [1], [0], [2], [0], [3], [1], [2], [3], [2]])
    data22 = np.array([[2], [0], [5], [1], [2], [2], [2], [2], [1], [2], [2], [1], [0], [0], [2]])
    data23 = np.array([[5], [2], [2], [5], [5], [2], [5], [4], [5], [2], [2], [2], [1], [5], [2]])
    data24 = np.array([[0], [2], [1], [0], [0], [0], [2], [2], [1], [3], [2], [1], [0], [5], [0]])
    data25 = np.array([[2], [2], [2], [1], [1], [0], [2], [0], [2], [2], [0], [0], [1], [1], [1]])
    data26 = np.array([[0], [0], [0], [0], [1], [0], [0], [0], [2], [0], [0], [4], [0], [0], [5]])
    data27 = np.array([[2], [4], [0], [1], [0], [0], [5], [0], [1], [0], [1], [0], [2], [2], [0]])
    data28 = np.array([[1], [1], [1], [0], [1], [2], [1], [1], [5], [0], [2], [1], [3], [0], [0]])
    data29 = np.array([[1], [1], [1], [1], [2], [2], [4], [2], [0], [0], [2], [1], [1], [3], [0]])
    data30 = np.array([[5], [5], [5], [4], [5], [1], [5], [4], [5], [4], [2], [5], [2], [1], [4]])
    data31 = np.array([[0], [0], [2], [1], [1], [1], [0], [0], [1], [0], [2], [0], [0], [5], [1]])
    data32 = np.array([[1], [1], [1], [1], [1], [2], [0], [0], [0], [2], [0], [0], [1], [2], [1]])
    data33 = np.array([[0], [0], [2], [2], [3], [1], [2], [1], [2], [1], [1], [0], [2], [0], [0]])
    data34 = np.array([[1], [2], [1], [1], [1], [1], [2], [0], [0], [0], [0], [1], [1], [0], [2]])
    data35 = np.array([[0], [1], [1], [5], [2], [1], [0], [0], [1], [1], [0], [2], [0], [2], [0]])
    data36 = np.array([[3], [0], [2], [2], [0], [2], [2], [1], [5], [1], [1], [2], [2], [0], [1]])
    data37 = np.array([[4], [5], [4], [1], [2], [4], [1], [5], [1], [2], [2], [5], [5], [1], [1]])
    data38 = np.array([[2], [1], [2], [0], [0], [0], [0], [0], [2], [0], [0], [2], [5], [1], [2]])
    data39 = np.array([[1], [3], [5], [1], [3], [1], [0], [2], [1], [0], [2], [2], [3], [3], [1]])
    data40 = np.array([[0], [1], [1], [3], [2], [0], [2], [0], [1], [1], [1], [1], [0], [3], [1]])
    data41 = np.array([[1], [0], [1], [0], [2], [2], [1], [5], [1], [5], [0], [1], [2], [2], [1]])
    data42 = np.array([[2], [3], [2], [2], [0], [0], [0], [1], [1], [0], [0], [0], [1], [5], [0]])
    data43 = np.array([[2], [5], [4], [5], [2], [2], [1], [5], [5], [4], [1], [5], [1], [5], [5]])
    state0 = np.array([1, 0, 0, 0, 3, 2, 2, 0, 1, 0, 2, 0, 0, 0, 0, ])
    state1 = np.array([1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ])
    state2 = np.array([0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 3, 0, ])
    state3 = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 3, ])
    state4 = np.array([3, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, ])
    state5 = np.array([1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, ])
    state6 = np.array([0, 0, 0, 1, 2, 0, 1, 1, 0, 1, 1, 3, 0, 0, 0, ])
    state7 = np.array([1, 0, 2, 1, 1, 0, 1, 3, 1, 1, 0, 0, 1, 0, 1, ])
    state8 = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1, ])
    state9 = np.array([0, 3, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, ])
    state10 = np.array([0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 2, ])
    state11 = np.array([0, 2, 0, 3, 1, 0, 0, 1, 1, 0, 1, 1, 0, 3, 1, ])
    state12 = np.array([0, 2, 1, 0, 1, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, ])
    state13 = np.array([3, 0, 2, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, ])
    state14 = np.array([0, 1, 0, 0, 1, 3, 0, 0, 1, 1, 1, 0, 1, 1, 0, ])
    state15 = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 3, ])
    state16 = np.array([0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, ])
    state17 = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, ])
    state18 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, ])
    state19 = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, ])
    state20 = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 3, 1, 1, 0, ])
    state21 = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 2, 1, ])
    state22 = np.array([1, 0, 3, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, ])
    state23 = np.array([3, 1, 1, 3, 3, 1, 3, 2, 3, 1, 1, 1, 0, 3, 1, ])
    state24 = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 0, 3, 0, ])
    state25 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, ])
    state26 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, ])
    state27 = np.array([1, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 0, ])
    state28 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 1, 0, 2, 0, 0, ])
    state29 = np.array([0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 0, 2, 0, ])
    state30 = np.array([3, 3, 3, 2, 3, 0, 3, 2, 3, 2, 1, 3, 1, 0, 2, ])
    state31 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, ])
    state32 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, ])
    state33 = np.array([0, 0, 1, 1, 2, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, ])
    state34 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, ])
    state35 = np.array([0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, ])
    state36 = np.array([2, 0, 1, 1, 0, 1, 1, 0, 3, 0, 0, 1, 1, 0, 0, ])
    state37 = np.array([2, 3, 2, 0, 1, 2, 0, 3, 0, 1, 1, 3, 3, 0, 0, ])
    state38 = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 3, 0, 1, ])
    state39 = np.array([0, 2, 3, 0, 2, 0, 0, 1, 0, 0, 1, 1, 2, 2, 0, ])
    state40 = np.array([0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, ])
    state41 = np.array([0, 0, 0, 0, 1, 1, 0, 3, 0, 3, 0, 0, 1, 1, 0, ])
    state42 = np.array([1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, ])
    state43 = np.array([1, 3, 2, 3, 1, 1, 0, 3, 3, 2, 0, 3, 0, 3, 3, ])



    #这里序列的长度是可以不固定的
    Datas = [data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,
             data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28,
             data29,data30,data31,data32,data33,data34,data35,data36,data37,data38,data39,data40,data41,data42,data43]
    # print Datas
    States =[state0,state1,state2,state3,state4,state5,state6,state7,state8,state9,state10,state11,state12,state13,state14,
             state15,state16,state17,state18,state19,state20,state21,state22,state23,state24,state25,state26,state27,state28,state29,
             state30,state31,state32,state33,state34,state35,state36,state37,state38,state39,state40,state41,state42,state43]
    return Datas,States


if __name__ == '__main__':
    Datas, States = prepareData_new()
    # 训练模型
    # datas = [
    #         [0,0,1,1,1,3,3,3,3,1,1,3,3,0,0,1,1,2,2,3],
    #         [3,3,3,3,0,1,1,1,1,2,2,0,0,3,3,3,1,1,0,0],
    #
    #         [1,1,1,1,3,2,2,0,0,2,1,1,1,2,2,2,0,2,2,0],
    #         [1,1,1,2,2,1,1,1,0,3,3,3,1,1,1,2,2,3,3,3],
    #
    #         [3,3,3,3,3,1,3,1,3,3,0,0,3,3,3,1,1,2,2,1],
    #         [3,3,3,1,1,1,1,1,3,3,2,2,0,0,3,3,1,1,1,3],
    #
    #         [1,1,1,1,3,3,2,1,1,1,2,3,3,0,3,1,2,1,1,2],
    #         [1,1,1,3,3,1,1,2,2,0,1,1,0,2,2,3,3,3,3,1],
    #         ]
    # data = [0,0,1,1,1,3,3,3,3,1,1,3,3,0,0,1,1,2,2,3]
    # # print np.array(datas)
    hmm = DiscreteHMM(4,6,300)
    # Data1 = np.array([[0],[0],[1],[1],[1],[3],[3],[3],[3],[1],[1],[3],[3],[0],[0],[1],[1],[2],[2],[3]])
    # Data2 = np.array([[3],[3],[3],[3],[0],[1],[1],[1],[1],[2],[2],[0],[0],[3],[3],[3],[1],[1],[0],[0]])
    # Data3 = np.array([[1],[1],[1],[1],[3],[2],[2],[0],[0],[2],[1],[1],[1],[2],[2],[2],[0],[2],[2],[0]])
    # Data4 = np.array([[1],[1],[1],[2],[2],[1],[1],[1],[0],[3],[3],[3],[1],[1],[1],[2],[2],[3],[3],[3]])
    # Data5 = np.array([[3],[3],[3],[3],[3],[1],[3],[1],[3],[3],[0],[0],[3],[3],[3],[1],[1],[2],[2],[1]])
    # Data6 = np.array([[3],[3],[3],[1],[1],[1],[1],[1],[3],[3],[2],[2],[0],[0],[3],[3],[1],[1],[1],[3]])
    # Data7 = np.array([[1],[1],[1],[1],[3],[3],[2],[1],[1],[1],[2],[3],[3],[0],[3],[1],[2],[1],[1],[2]])
    # Data8 = np.array([[1],[1],[1],[3],[3],[1],[1],[2],[2],[0],[1],[1],[0],[2],[2],[3],[3],[3],[3],[1]])
    # Datas = [Data1,Data2,Data3,Data4,Data5,Data6,Data7,Data8]
    # print Datas
    # # state1 = np.array([[0]])
    # # state2 = np.array([[1]])
    # # state3 = np.array([[2]])
    # # state4 = np.array([[3]])
    # # states = [state1,state2,state3,state4]
    # # print states
    # # hmm.train_batch(Datas,states)
    hmm.train_batch(Datas,States)
    print hmm.PI
    print hmm.A
    print hmm.B

    # # hmm.train(np.array(datas))
    #
    # # A = np.array([[0.67207095, 0.00194653, 0.12922411, 0.19675841],
    # # [0.00286666 ,0.47891329, 0.51551516 ,0.00270489],
    # # [0.21869559 ,0.00102436 ,0.64688482, 0.13339523],
    # # [0.04601312 ,0.46594308 ,0.00154478 ,0.48649902]])
    # # B = np.array([[0.28004988 ,0.04772025 ,0.67561186 ,0.04943152],
    # # [0.20888691 ,0.07863687, 0.04272959, 0.74159338],
    # # [0.01638039, 0.94583511 ,0.03985226 ,0.0311495 ],
    # # [0.16411452 ,0.03096597, 0.02424079, 0.83935692]])
    # # pi = np.array([0.00320128 ,0.00601686, 0.50248361 ,0.48829825])
    # # print A
    # # print B
    # # print pi
    #
    # testData = [0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 2, 0, 2, 0, 3, 0, 0, 1, 0, 3]
    # o = np.array([[0], [0], [3], [0], [0], [3], [1], [1], [2], [0], [2], [0], [2], [0], [3], [0], [0], [1], [0], [3]])
    o = np.array([[1], [4], [2], [5], [5], [4], [4], [2], [5], [5], [4], [4]])

    print hmm.decode(o)
