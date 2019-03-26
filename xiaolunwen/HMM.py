#coding=utf8

import time
import copy
import numpy as np

class HMM:
    def __init__(self, Ann, Bnm, Pi, O):
        self.A = np.array(Ann, np.float)
        self.B = np.array(Bnm, np.float)
        self.Pi = np.array(Pi, np.float)
        # self.O = np.array(O, np.float)
        self.O = np.array(O)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
        np.set_printoptions(precision=6)

    # 这个函数完全是参照李航的书来的
    def viterbi(self):
        # given O,lambda .finding I
        # 个观测序列长度
        T = len(self.O)
        # 初始化返回结果
        I = np.zeros(T, np.int)
        # 初始化δ矩阵
        delta = np.zeros((T, self.N), np.float)
        # 初始化ψ矩阵
        psi = np.zeros((T, self.N), np.float)

        for i in range(self.N):
            # pi*b
            delta[0, i] = self.Pi[i] * self.B[i, self.O[0]]
            # delta[0, i] = self.B[i, self.O[0]] * self.Pi[i]
            # 第一次递推的时候ψ等于0
            psi[0, i] = 0
        # 递推
        for t in range(1, T):
            for i in range(self.N):
                delta[t, i] = self.B[i, self.O[t]] * np.array([delta[t - 1, j] * self.A[j, i]
                                                               for j in range(self.N)]).max()
                psi[t, i] = np.array([delta[t - 1, j] * self.A[j, i]
                                      for j in range(self.N)]).argmax()

        P_T = delta[T - 1, :].max()
        I[T - 1] = delta[T - 1, :].argmax()

        for t in range(T - 2, -1, -1):
            I[t] = psi[t + 1, I[t + 1]]

        return I

    def forward(self):
        # 时间T（一共有多少时刻）
        T = len(self.O)
        # 创建α变量，T对应时间，N对应状态数
        alpha = np.zeros((T, self.N), np.float)
        # 初始化α变量，t=0时，这里的alpha[0, i]中的0和self.O[0]中的0是对应的
        for i in range(self.N):
            alpha[0, i] = self.Pi[i] * self.B[i, self.O[0]]
            # alpha[0, i] = self.B[i, self.O[0]] * self.Pi[i]
        # 递推

        for t in range(T - 1):
            for i in range(self.N):
                # 从t时到t+1时刻所有可能转移到状态i求和
                summation = 0  # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += alpha[t, j] * self.A[j, i]
                # alpha这里的t+1和后面的O[t+1]是对应的
                alpha[t + 1, i] = summation * self.B[i, self.O[t + 1]]

        summation = 0.0
        for i in range(self.N):
            summation += alpha[T - 1, i]
        Polambda = summation
        return Polambda, alpha

    def backward(self):
        # 时间T（一共有多少时刻）
        T = len(self.O)
        # 创建β变量，T对应时间，N对应状态数
        beta = np.zeros((T, self.N), np.float)
        # 初始化β变量
        for i in range(self.N):
            beta[T - 1, i] = 1.0
        # 递推
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                summation = 0.0  # for every i 'summation' should reset to '0'
                for j in range(self.N):
                    summation += self.A[i, j] * self.B[j, self.O[t + 1]] * beta[t + 1, j]
                beta[t, i] = summation

        Polambda = 0.0
        for i in range(self.N):
            Polambda += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]
        return Polambda, beta

    # 计算中间变量γ ，γti在时刻t处于状态qi的概率
    def compute_gamma(self, alpha, beta):
        # 时间T（一共有多少时刻）
        T = len(self.O)
        # 创建γ变量，T对应时间，N对应状态数
        gamma = np.zeros((T, self.N), np.float)  # the probability of Ot=q
        # 计算每一项
        for t in range(T):
            for i in range(self.N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / sum(
                    alpha[t, j] * beta[t, j] for j in range(self.N))
        return gamma

    # 计算中间变量ζ ，ζtij在时刻t处于状态qi且在时刻t+1处于状态qj的概率
    def compute_xi(self, alpha, beta):
        # 时间T（一共有多少时刻）
        T = len(self.O)
        # 创建ζ变量，是一个三维的变量
        xi = np.zeros((T - 1, self.N, self.N), np.float)  # note that: not T
        # 计算每一项
        for t in range(T - 1):  # note: not T
            for i in range(self.N):
                for j in range(self.N):
                    numerator = alpha[t, i] * self.A[i, j] * self.B[j, self.O[t + 1]] * beta[t + 1, j]
                    # the multiply term below should not be replaced by 'nummerator'，
                    # since the 'i,j' in 'numerator' are fixed.
                    # In addition, should not use 'i,j' below, to avoid error and confusion.
                    denominator = sum(sum(
                        alpha[t, i1] * self.A[i1, j1] * self.B[j1, self.O[t + 1]] * beta[t + 1, j1]
                        for j1 in range(self.N))  # the second sum
                            for i1 in range(self.N))  # the first sum
                    xi[t, i, j] = numerator / denominator
        return xi

    def Baum_Welch(self):
        # 时间T（一共有多少时刻）
        T = len(self.O)
        V = [k for k in range(self.M)]

        # initialization - lambda，模型训练的初始化参数，自己手动设定
        self.A = np.array(([[0, 1, 0, 0], [0.4, 0, 0.6, 0], [0, 0.4, 0, 0.6], [0, 0, 0.5, 0.5]]), np.float)
        self.B = np.array(([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]]), np.float)

        # mean value may not be a good choice，平均值
        self.Pi = np.array(([1.0 / self.N] * self.N), np.float)  # must be 1.0 , if 1/3 will be 0
        # self.A = np.array([[1.0 / self.N] * self.N] * self.N) # must array back, then can use[i,j]
        # self.B = np.array([[1.0 / self.M] * self.M] * self.N)

        x = 1
        delta_lambda = x + 1
        times = 0
        # iteration - lambda
        while delta_lambda > x:  # x
            # 前向计算α
            Polambda1, alpha = self.forward()  # get alpha
            # 后向计算β
            Polambda2, beta = self.backward()  # get beta
            # 计算中间变量
            gamma = self.compute_gamma(alpha, beta)  # use alpha, beta
            xi = self.compute_xi(alpha, beta)
            # 第n次的值
            # lambda_n = [self.A, self.B, self.Pi]
            lambda_n = copy.deepcopy([self.A, self.B, self.Pi])
            # 计算Aij
            for i in range(self.N):
                for j in range(self.N):
                    numerator = sum(xi[t, i, j] for t in range(T - 1))
                    denominator = sum(gamma[t, i] for t in range(T - 1))
                    self.A[i, j] = numerator / denominator
            # 计算Bjk
            for j in range(self.N):
                for k in range(self.M):
                    numerator = sum(gamma[t, j] for t in range(T) if self.O[t] == V[k])  # TBD
                    denominator = sum(gamma[t, j] for t in range(T))
                    self.B[j, k] = numerator / denominator
            # 计算Pi
            for i in range(self.N):
                self.Pi[i] = gamma[0, i]

            # if sum directly, there will be positive and negative offset
            delta_A = map(abs, lambda_n[0] - self.A)  # delta_A is still a matrix
            delta_B = map(abs, lambda_n[1] - self.B)
            delta_Pi = map(abs, lambda_n[2] - self.Pi)
            delta_lambda = sum([sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi)])
            times += 1
            # print times

        return self.A, self.B, self.Pi

    def forward_with_scale(self):
        T = len(self.O)
        alpha_raw = np.zeros((T, self.N), np.float)
        alpha = np.zeros((T, self.N), np.float)
        c = [i for i in range(T)]  # scaling factor; 0 or sequence doesn't matter

        for i in range(self.N):
            # print alpha_raw[0, i]
            # print self.Pi[i]
            # print self.O[0]
            # print self.B[i, self.O[0]]
            alpha_raw[0, i] = self.Pi[i] * self.B[i, self.O[0]]

        c[0] = 1.0 / sum(alpha_raw[0, i] for i in range(self.N))
        for i in range(self.N):
            alpha[0, i] = c[0] * alpha_raw[0, i]

        for t in range(T - 1):
            for i in range(self.N):
                summation = 0.0
                for j in range(self.N):
                    summation += alpha[t, j] * self.A[j, i]
                alpha_raw[t + 1, i] = summation * self.B[i, self.O[t + 1]]

            c[t + 1] = 1.0 / sum(alpha_raw[t + 1, i1] for i1 in range(self.N))

            for i in range(self.N):
                alpha[t + 1, i] = c[t + 1] * alpha_raw[t + 1, i]

        return alpha, c

    def backward_with_scale(self, c):
        T = len(self.O)
        beta_raw = np.zeros((T, self.N), np.float)
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta_raw[T - 1, i] = 1.0
            beta[T - 1, i] = c[T - 1] * beta_raw[T - 1, i]

        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                summation = 0.0
                for j in range(self.N):
                    summation += self.A[i, j] * self.B[j, self.O[t + 1]] * beta[t + 1, j]
                beta[t, i] = c[t] * summation  # summation = beta_raw[t,i]

        return beta

    def Baum_Welch_with_scale(self):
        T = len(self.O)
        V = [k for k in range(self.M)]

        # initialization - lambda   ,  should be float(need .0)
        self.A = np.array([[0.2, 0.2, 0.3, 0.3], [0.2, 0.1, 0.6, 0.1], [0.3, 0.4, 0.1, 0.2], [0.3, 0.2, 0.2, 0.3]])
        self.B = np.array([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]])

        x = 5
        delta_lambda = x + 1
        times = 0
        # iteration - lambda
        while delta_lambda > x:  # x
            alpha, c = self.forward_with_scale()
            beta = self.backward_with_scale(c)

            # lambda_n = [self.A, self.B, self.Pi]
            lambda_n = copy.deepcopy([self.A, self.B, self.Pi])

            for i in range(self.N):
                for j in range(self.N):
                    numerator_A = sum(alpha[t, i] * self.A[i, j] * self.B[j, self.O[t + 1]]
                                      * beta[t + 1, j] for t in range(T - 1))
                    denominator_A = sum(alpha[t, i] * beta[t, i] / c[t] for t in range(T - 1))
                    self.A[i, j] = numerator_A / denominator_A

            for j in range(self.N):
                for k in range(self.M):
                    numerator_B = sum(alpha[t, j] * beta[t, j] / c[t]
                                      for t in range(T) if self.O[t] == V[k])  # TBD
                    denominator_B = sum(alpha[t, j] * beta[t, j] / c[t] for t in range(T))
                    self.B[j, k] = numerator_B / denominator_B

            # Pi have no business with c
            denominator_Pi = sum(alpha[0, j] * beta[0, j] for j in range(self.N))
            for i in range(self.N):
                self.Pi[i] = alpha[0, i] * beta[0, i] / denominator_Pi
                # self.Pi[i] = gamma[0,i]

            # if sum directly, there will be positive and negative offset
            delta_A = map(abs, lambda_n[0] - self.A)  # delta_A is still a matrix
            delta_B = map(abs, lambda_n[1] - self.B)
            delta_Pi = map(abs, lambda_n[2] - self.Pi)
            delta_lambda = sum([sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi)])

            times += 1
            # print times
        return self.A, self.B, self.Pi

# for multiple sequences of observations symbols(with scaling alpha & beta)
# out of class HMM, independent function
# 同时输入多个观测序列的对应实现
def modified_Baum_Welch_with_scale(O_set):
    # initialization - lambda
    A = np.array([[0.2, 0.2, 0.3, 0.3], [0.2, 0.1, 0.6, 0.1], [0.3, 0.4, 0.1, 0.2], [0.3, 0.2, 0.2, 0.3]])
    B = np.array([[0.2, 0.2, 0.3, 0.3], [0.2, 0.1, 0.6, 0.1], [0.3, 0.4, 0.1, 0.2], [0.3, 0.2, 0.2, 0.3]])
    # B = np.array([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]])
    Pi = np.array([0.05,0.05,0.45,0.45])

    # computing alpha_set, beta_set, c_set
    O_length = len(O_set)
    whatever = [j for j in range(O_length)]
    alpha_set, beta_set = whatever, whatever
    c_set = [j for j in range(O_length)]  # can't use whatever, the c_set will be 3d-array ???

    N = A.shape[0]
    M = B.shape[1]
    T = [j for j in range(O_length)]  # can't use whatever, the beta_set will be 1d-array ???
    for i in range(O_length):
        T[i] = len(O_set[i])
    V = [k for k in range(M)]

    x = 1
    delta_lambda = x + 1
    times = 0
    delta_lambdas = []
    while delta_lambda > x:  # iteration - lambda
        lambda_n = copy.deepcopy([A, B])
        for i in range(O_length):
            alpha_set[i], c_set[i] = HMM(A, B, Pi, O_set[i]).forward_with_scale()
            beta_set[i] = HMM(A, B, Pi, O_set[i]).backward_with_scale(c_set[i])
            # alpha_set[i], c_set[i] = HMM(A, B, Pi, O_set[i]).forward()
            # beta_set[i] = HMM(A, B, Pi, O_set[i]).backward(c_set[i])

        for i in range(N):
            for j in range(N):

                numerator_A = 0.0
                denominator_A = 0.0
                for l in range(O_length):
                    raw_numerator_A = sum(alpha_set[l][t, i] * A[i, j] * B[j, O_set[l][t + 1]]
                                          * beta_set[l][t + 1, j] for t in range(T[l] - 1))
                    numerator_A += raw_numerator_A

                    raw_denominator_A = sum(alpha_set[l][t, i] * beta_set[l][t, i] / c_set[l][t]
                                            for t in range(T[l] - 1))
                    denominator_A += raw_denominator_A

                A[i, j] = numerator_A / denominator_A

        for j in range(N):
            for k in range(M):

                numerator_B = 0.0
                denominator_B = 0.0
                for l in range(O_length):
                    raw_numerator_B = sum(alpha_set[l][t, j] * beta_set[l][t, j]
                                          / c_set[l][t] for t in range(T[l]) if O_set[l][t] == V[k])
                    numerator_B += raw_numerator_B

                    raw_denominator_B = sum(alpha_set[l][t, j] * beta_set[l][t, j]
                                            / c_set[l][t] for t in range(T[l]))
                    denominator_B += raw_denominator_B
                B[j, k] = numerator_B / denominator_B

        # Pi should not need to computing in this case,
        # in other cases, will get some corresponding Pi

        # if sum directly, there will be positive and negative offset
        delta_A = map(abs, lambda_n[0] - A)  # delta_A is still a matrix
        delta_B = map(abs, lambda_n[1] - B)
        delta_lambda = sum([sum(sum(delta_A)), sum(sum(delta_B))])
        # if delta_lambda not in delta_lambdas:
        #     delta_lambdas.append(delta_lambda)
        # else:
        #     # 有差值在列表里面说明现在可能是两边跳转的情形了，此时也要跳出循环
        #     break
        times += 1
        # if times%100==0:
        print times
        print delta_lambda
        print A
        print B
        print Pi
        # time.sleep(0.5)
        # print times

    return A, B

if __name__ == '__main__':
    # 训练模型
    data = [
            [0,0,1,1,1,3,3,3,3,1,1,3,3,0,0,1,1,2,2,3],
            [3,3,3,3,0,1,1,1,1,2,2,0,0,3,3,3,1,1,0,0],

            [1,1,1,1,3,2,2,0,0,2,1,1,1,2,2,2,0,2,2,0],
            [1,1,1,2,2,1,1,1,0,3,3,3,1,1,1,2,2,3,3,3],

            [3,3,3,3,3,1,3,1,3,3,0,0,3,3,3,1,1,2,2,1],
            [3,3,3,1,1,1,1,1,3,3,2,2,0,0,3,3,1,1,1,3],

            [1,1,1,1,3,3,2,1,1,1,2,3,3,0,3,1,2,1,1,2],
            [1,1,1,3,3,1,1,2,2,0,1,1,0,2,2,3,3,3,3,1],
            ]
    modified_Baum_Welch_with_scale(data)

    # 测试模型
    A = np.array([[0.20552862,0.14043902,0.34331828,0.31071408],
                [0.17942047,0.05644943,0.67232055,0.09180956],
                [0.32971145 ,0.34677507, 0.10887219 ,0.21464129],
                [0.31486195 ,0.14578459 ,0.22394397 ,0.31540948]])
    B = np.array([[0.26767989 ,0.22750701 ,0.25974235 ,0.24507075],
                [0.26767989, 0.22750701, 0.25974235, 0.24507075],
                [0.26767989 ,0.22750701 ,0.25974235 ,0.24507075],
                [0.26767989, 0.22750701, 0.25974235, 0.24507075]])
    pi = np.array([0.05, 0.05, 0.45 ,0.45])
    testData = [0,0,0,0,0,0,1,1,2,0,2,0,2,0,3,0,0,1,0,3]
    hmm = HMM(A, B, pi ,testData)
    I = hmm.viterbi()
    print I