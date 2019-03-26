# -*- coding:utf-8 -*-
# 测试离散隐马尔科夫链模型
# 引入一个经典的HMM库 hmmlearn作为比较
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq
import unittest
import hmm
import hmmlearn.hmm
import numpy as np
from math import sqrt,log



class ContrastHMM():
    def __init__(self, n_state, n_feature):
        self.module = hmmlearn.hmm.MultinomialHMM(n_components=n_state)
        # 初始概率
        self.module.startprob_ = np.random.random(n_state)
        self.module.startprob_ = self.module.startprob_ / np.sum(self.module.startprob_)
        # print self.module.startprob_
        # 转换概率
        self.module.transmat_ = np.random.random((n_state,n_state))
        self.module.transmat_ = self.module.transmat_ / np.repeat(np.sum(self.module.transmat_, 1),n_state).reshape((n_state,n_state))
        # print self.module.transmat_
        # 发射概率
        self.module.emissionprob_ = np.random.random(size=(n_state,n_feature))
        self.module.emissionprob_ = self.module.emissionprob_ / np.repeat(np.sum(self.module.emissionprob_, 1),n_feature).reshape((n_state,n_feature))
        # print self.module.emissionprob_

# 计算平方误差
def s_error(A, B):
    return sqrt(np.sum((A-B)*(A-B)))/np.sum(B)

class DiscreteHMM_Test(unittest.TestCase):

    def setUp(self):
        # 建立两个HMM，隐藏状态个数为4，X可能分布为10类
        n_state =4
        n_feature = 10
        X_length = 2
        n_batch = 100 # 批量数目
        self.n_batch = n_batch
        self.X_length = X_length
        self.test_hmm = hmm.DiscreteHMM(n_state, n_feature)
        self.comp_hmm = ContrastHMM(n_state, n_feature)
        self.X, self.Z = self.comp_hmm.module.sample(self.X_length*10)
        # print self.X
        # print type(self.X)
        # print self.Z
        # print type(self.Z)
        self.test_hmm.train(self.X, self.Z)

    def test_train_batch(self):
        X = []
        Z = []
        for b in range(self.n_batch):
            b_X, b_Z = self.comp_hmm.module.sample(self.X_length)
            X.append(b_X)
            Z.append(b_Z)
        print X
        print type(X)

        batch_hmm = hmm.DiscreteHMM(self.test_hmm.n_state, self.test_hmm.x_num)
        batch_hmm.train_batch(X, Z)
        # 判断概率参数是否接近
        # 初始概率判定没有通过！！！
        self.assertAlmostEqual(s_error(batch_hmm.start_prob, self.comp_hmm.module.startprob_), 0, 1)
        self.assertAlmostEqual(s_error(batch_hmm.transmat_prob, self.comp_hmm.module.transmat_), 0, 1)
        self.assertAlmostEqual(s_error(batch_hmm.emission_prob, self.comp_hmm.module.emissionprob_), 0, 1)

    def test_train(self):
        # 判断概率参数是否接近
        # 单批量的初始概率一定是不准的
        # self.assertAlmostEqual(s_error(self.test_hmm.start_prob, self.comp_hmm.module.startprob_), 0, 1)
        self.assertAlmostEqual(s_error(self.test_hmm.transmat_prob, self.comp_hmm.module.transmat_), 0, 1)
        self.assertAlmostEqual(s_error(self.test_hmm.emission_prob, self.comp_hmm.module.emissionprob_), 0, 1)

    def test_X_prob(self):
        X,_ = self.comp_hmm.module.sample(self.X_length)
        prob_test = self.test_hmm.X_prob(X)
        prob_comp = self.comp_hmm.module.score(X)
        self.assertAlmostEqual(s_error(prob_test, prob_comp), 0, 1)

    def test_predict(self):
        X, _ = self.comp_hmm.module.sample(self.X_length)
        prob_next = self.test_hmm.predict(X,np.random.randint(0,self.test_hmm.x_num-1))
        self.assertEqual(prob_next.shape,(self.test_hmm.n_state,))

    def test_decode(self):
        X,_ = self.comp_hmm.module.sample(self.X_length)
        test_decode = self.test_hmm.decode(X)
        _, comp_decode = self.comp_hmm.module.decode(X)
        self.assertAlmostEqual(s_error(test_decode, comp_decode), 0, 1)

if __name__ == '__main__':
    unittest.main()