# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:12:49 2017

@author: KAWALAB
"""


import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer.datasets import get_mnist
from chainer.dataset import concat_examples

def load_mnist_as_ndarray():
    train, test = get_mnist(ndim=3) #MNISTの取得
    train = concat_examples(train) #numpy配列への変換
    test = concat_examples(test) #numpy配列への変換
    return train, test

class ConvNet(chainer.Chain):
    def __init__(self):
        super(ConvNet, self).__init__(
            conv1=L.Convolution2D(1, 10, 3),  # 28 -> 26
            conv2=L.Convolution2D(10, 10, 4),  # 13 -> 10
            fc1=L.Linear(250, 10)
        )
        
    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, 2)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, 2)
        y = self.fc1(h)
        return y
    
#この.pyファイルをベースに実行した場合は__name__に__main__が入る
if __name__ == '__main__': 
    num_epochs = 100
    batch_size = 500
    learning_rate = 0.001
    
    #データの読み込み
    train, test = load_mnist_as_ndarray()
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)
    
    #モデルとオプティマイザの準備
    model = ConvNet()
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)
    
    #訓練ループ
    loss_log = []
    for epoch in range(num_epochs):
        for i in range(0, num_train, batch_size):
            x_batch = x_train[i:i+batch_size]
            c_batch = c_train[i:i+batch_size]
            y_batch = model(x_batch)
            
            loss = F.softmax_cross_entropy(y_batch, c_batch)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            
            accuracy = F.accuracy(y_batch, c_batch)
            print(epoch, accuracy.data, loss.data)
            loss_log.append(loss.data)
            
        plt.plot(loss_log)
        plt.show()
            