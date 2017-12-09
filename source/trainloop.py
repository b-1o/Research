#coding:utf-8
import train100
import numpy as np

def pattern1():
    # カテゴリごとのサンプル数
    sample_num = np.array([4000,4000,4000,4000,4000,4000,4000,4000,4000,4000])

    train100.train(sample_num)


def pattern2():
    # カテゴリごとのサンプル数
    sample_num = np.array([4000,4000,4000,4000,4000,4000,4000,4000,4000,4000])

    train100.train(sample_num)

    for i in range(10):

        sample_num[i] = 20
        train100.train(sample_num)

        sample_num[i] = 50
        train100.train(sample_num)

        sample_num[i] = 100
        train100.train(sample_num)

        sample_num[i] = 250
        train100.train(sample_num)

        sample_num[i] = 500
        train100.train(sample_num)

        sample_num[i] = 1000
        train100.train(sample_num)

        sample_num[i] = 4000



if __name__ == "__main__":
    pattern2()
