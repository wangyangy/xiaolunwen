# -*- coding:utf-8 -*-

from hmmlearn import hmm
import numpy as np

def compute():
    states = ["box 1", "box 2", "box3"]
    n_states = len(states)

    observations = ["red", "white"]
    n_observations = len(observations)

    start_probability = np.array([0.2, 0.4, 0.4])

    transition_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])

    model = hmm.MultinomialHMM(n_components=n_states)
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    seen = np.array([[0, 1, 0]]).T
    logprob, box = model.decode(seen, algorithm="viterbi")
    print "观测序列：",
    for i in range(len(seen)):
        print observations[int(seen[i])],
    print
    print "最有可能隐藏状态：",
    for i in range(len(seen)):
        print states[int(box[i])],
    print
    print "观测序列：",
    box2 = model.predict(seen)
    for i in range(len(seen)):
        print observations[int(seen[i])],
    print
    print "最有可能隐藏状态：",
    for i in range(len(seen)):
        print states[int(box2[i])],
    print
    # 要注意的是score函数返回的是以自然对数为底的对数概率值，我们在HMM问题一中手动计算的结果是未取对数的原始概率是0.13022。对比一下：
    print "分数：",
    print model.score(seen)


def xuexicanshu():
    states = ["box 1", "box 2", "box3"]
    n_states = len(states)

    observations = ["red", "white"]
    n_observations = len(observations)
    model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
    X2 = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 1, 1]])
    print X2
    model2.fit(X2)
    print model2.startprob_
    print model2.transmat_
    print model2.emissionprob_
    print model2.score(X2)
    model2.fit(X2)
    print model2.startprob_
    print model2.transmat_
    print model2.emissionprob_
    print model2.score(X2)
    model2.fit(X2)
    print model2.startprob_
    print model2.transmat_
    print model2.emissionprob_
    print model2.score(X2)


def myTest():
    # N:s1(异常，活跃),s2（异常，不活跃）,s3（正常，活跃）,s4（正常，不活跃）
    states = ["s1", "s2", "s3", "s4"]
    n_states = len(states)
    # y0: 风险访问，活跃
    # y1: 正常访问，不活跃
    # y2: 风险访问，不活跃
    # y3: 正常访问，活跃
    observations = ["y0", "y1","y2","y3"]
    n_observations = len(observations)

    #[s1:0.15, s4:0.25, s2:0.1, s3:0.5]
    start_probability = np.array([0.15, 0.1, 0.5,0.25])

    remodel = hmm.MultinomialHMM(n_components=n_states, n_iter=1000)
    remodel.startprob_ = start_probability
    X = [
        [3, 3, 3, 1, 3, 1, 3, 3, 3, 1, 2, 3, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 1, 2, 3, 1, 1, 3, 2, 3],
        [3, 3, 3, 1, 1, 2, 3, 1, 1, 1, 3, 3, 2, 3, 3],
        [2, 3, 3, 3, 1, 1, 3, 3, 3, 1, 3, 1, 3, 3, 3],
        [3, 3, 3, 1, 1, 3, 1, 3, 1, 1, 1, 3, 3, 3, 2],
        [3, 3, 1, 1, 1, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3],
        [1, 3, 1, 3, 3, 3, 3, 1, 2, 2, 3, 3, 1, 3, 3],
        [1, 3, 3, 3, 3, 1, 3, 3, 1, 1, 3, 3, 3, 2, 3],
        [3, 1, 1, 1, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 1],
        [3, 3, 1, 1, 3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 2],
        [3, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 2, 3],
        [1, 3, 1, 3, 3, 3, 2, 2, 3, 3, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 2, 3, 3],
        [1, 3, 3, 3, 1, 3, 1, 3, 2, 3, 2, 3, 3, 3, 1],
        [3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 2, 1, 3, 3],
        [3, 1, 1, 1, 3, 3, 3, 3, 3, 2, 3, 2, 3, 1, 3],
        [3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3, 2],
        [3, 3, 3, 1, 3, 3, 1, 3, 1, 0, 3, 3, 1, 2, 3],
        [3, 3, 3, 0, 3, 1, 1, 3, 3, 2, 3, 3, 2, 3, 3],
        [3, 3, 0, 2, 3, 3, 2, 3, 3, 1, 1, 3, 3, 3, 1],
        [1, 1, 1, 1, 2, 3, 3, 1, 1, 3, 1, 3, 1, 1, 1],
        [1, 3, 1, 3, 1, 1, 1, 3, 1, 1, 2, 1, 1, 3, 1],
        [1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 2, 1, 1, 1],
        [1, 1, 3, 2, 1, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1],
        [3, 1, 1, 1, 1, 2, 1, 1, 2, 1, 3, 3, 1, 1, 1],
        [1, 1, 1, 3, 3, 1, 1, 3, 1, 2, 2, 1, 1, 3, 1],
        [1, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 1],
        [1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 3, 1],
        [1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 3, 1, 1, 3, 1],
        [1, 1, 3, 3, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 3],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 2, 2, 2, 0, 0, 0, 2, 1, 0],
        [0, 0, 0, 2, 0, 2, 0, 0, 1, 0, 2, 0, 2, 0, 0],
        [0, 2, 0, 2, 0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0],
        [2, 2, 2, 0, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 0],
        [2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2],
        [2, 2, 0, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1],
        [2, 1, 0, 1, 2, 2, 2, 0, 2, 1, 2, 1, 2, 0, 1],
    ]
    # print np.array(X)
    for i in range(10):
        remodel.fit(np.array(X))
        print remodel.startprob_
        print remodel.transmat_
        print remodel.emissionprob_
        print remodel.score(X)

        data = [[0, 0, 1, 0, 0, 0, 1, 1, 2, 0, 2, 0, 2, 0, 3]]
        data = np.array(data).T
        logprob, state  = remodel.decode(np.array(data), algorithm="viterbi")
        print "观测序列：",
        for i in range(len(data)):
            print observations[int(data[i])],
        print
        print "最有可能隐藏状态：",
        for i in range(len(data)):
            print states[int(state[i])],
        print


def myTestNew():
    # N:s1(异常，活跃),s2（异常，不活跃）,s3（正常，活跃）,s4（正常，不活跃）
    states = ["s1", "s2", "s3", "s4"]
    # states = ["s1", "s2"]
    n_states = len(states)
    # y0: 风险访问，活跃
    # y1: 正常访问，不活跃
    # y2: 风险访问，不活跃
    # y3: 正常访问，活跃
    observations = ["o1", "o2","o3","o4","o5","o6"]
    n_observations = len(observations)

    #[s1:0.15, s4:0.25, s2:0.1, s3:0.5]
    # start_probability = np.array([0.15, 0.1, 0.5,0.25])

    remodel = hmm.MultinomialHMM(n_components=n_states, n_iter=300)
    # 手动设置初始状态概率
    # remodel.startprob_ = start_probability

    X = [
        [2, 0, 0, 1, 5, 4, 4, 1, 2, 1, 3, 0, 0, 0, 0],
        [2, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 2, 1, 1, 0, 3, 0, 2, 1, 1, 1, 5, 0],
        [1, 1, 0, 2, 2, 1, 0, 1, 1, 2, 2, 0, 2, 2, 5],
        [5, 0, 1, 0, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 0],
        [2, 2, 2, 2, 2, 3, 2, 2, 2, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 2, 4, 0, 2, 2, 0, 2, 2, 5, 1, 1, 1],
        [2, 1, 3, 2, 2, 0, 2, 5, 2, 2, 0, 1, 2, 0, 2],
        [1, 1, 0, 1, 0, 2, 0, 2, 2, 1, 3, 0, 2, 0, 2],
        [0, 5, 0, 0, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0],
        [1, 1, 0, 2, 0, 3, 0, 0, 0, 2, 0, 0, 2, 0, 4],
        [1, 3, 0, 5, 2, 1, 0, 2, 2, 0, 2, 2, 1, 5, 2],
        [0, 3, 2, 1, 2, 1, 0, 2, 2, 2, 3, 3, 1, 0, 1],
        [5, 0, 4, 0, 2, 0, 0, 2, 1, 3, 1, 1, 1, 2, 1],
        [1, 2, 1, 1, 2, 5, 1, 0, 2, 2, 2, 0, 2, 2, 1],
        [0, 1, 0, 1, 2, 2, 0, 1, 2, 1, 0, 0, 0, 2, 5],
        [0, 2, 4, 0, 1, 2, 0, 2, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 2, 2, 1, 0, 1, 0, 0, 1, 0, 4, 1],
        [2, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 3, 1, 2],
        [1, 2, 0, 0, 0, 2, 1, 2, 0, 2, 2, 2, 0, 2, 0],
        [2, 1, 2, 0, 1, 1, 2, 0, 0, 0, 2, 5, 2, 2, 1],
        [2, 0, 2, 2, 1, 1, 1, 0, 2, 0, 3, 1, 2, 3, 2],
        [2, 0, 5, 1, 2, 2, 2, 2, 1, 2, 2, 1, 0, 0, 2],
        [5, 2, 2, 5, 5, 2, 5, 4, 5, 2, 2, 2, 1, 5, 2],
        [0, 2, 1, 0, 0, 0, 2, 2, 1, 3, 2, 1, 0, 5, 0],
        [2, 2, 2, 1, 1, 0, 2, 0, 2, 2, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 4, 0, 0, 5],
        [2, 4, 0, 1, 0, 0, 5, 0, 1, 0, 1, 0, 2, 2, 0],
        [1, 1, 1, 0, 1, 2, 1, 1, 5, 0, 2, 1, 3, 0, 0],
        [1, 1, 1, 1, 2, 2, 4, 2, 0, 0, 2, 1, 1, 3, 0],
        [5, 5, 5, 4, 5, 1, 5, 4, 5, 4, 2, 5, 2, 1, 4],
        [0, 0, 2, 1, 1, 1, 0, 0, 1, 0, 2, 0, 0, 5, 1],
        [1, 1, 1, 1, 1, 2, 0, 0, 0, 2, 0, 0, 1, 2, 1],
        [0, 0, 2, 2, 3, 1, 2, 1, 2, 1, 1, 0, 2, 0, 0],
        [1, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 0, 2],
        [0, 1, 1, 5, 2, 1, 0, 0, 1, 1, 0, 2, 0, 2, 0],
        [3, 0, 2, 2, 0, 2, 2, 1, 5, 1, 1, 2, 2, 0, 1],
        [4, 5, 4, 1, 2, 4, 1, 5, 1, 2, 2, 5, 5, 1, 1],
        [2, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 5, 1, 2],
        [1, 3, 5, 1, 3, 1, 0, 2, 1, 0, 2, 2, 3, 3, 1],
        [0, 1, 1, 3, 2, 0, 2, 0, 1, 1, 1, 1, 0, 3, 1],
        [1, 0, 1, 0, 2, 2, 1, 5, 1, 5, 0, 1, 2, 2, 1],
        [2, 3, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 1, 5, 0],
        [2, 5, 4, 5, 2, 2, 1, 5, 5, 4, 1, 5, 1, 5, 5],
    ]
    # print np.array(X)
    for i in range(10):
        remodel.fit(np.array(X))
        print remodel.startprob_
        print remodel.transmat_
        print remodel.emissionprob_
        print remodel.score(X)

        data = [[ 1, 4, 2, 5, 5, 4, 4, 2, 5, 5, 4, 4]]
        data = np.array(data).T
        logprob, state  = remodel.decode(np.array(data), algorithm="viterbi")
        print "观测序列：",
        for i in range(len(data)):
            print observations[int(data[i])],
        print
        print "最有可能隐藏状态：",
        for i in range(len(data)):
            print states[int(state[i])],
        print "----------------------"


if __name__=="__main__":
    # compute()
    # xuexicanshu()
    myTestNew()