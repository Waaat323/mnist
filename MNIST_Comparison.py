# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:46:53 2017

@author: kawalab
"""
#Hello wold
import argparse


import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda
from chainer import optimizers
from chainer.datasets import get_mnist
from chainer.dataset import concat_examples


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default= 0, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np



def load_Mnist():
    train, test = get_mnist(ndim=3)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test

class ConvNet(chainer.Chain):
    def __init__(self):
        super(ConvNet,self).__init__(
            conv1 = L.Convolution2D(1, 10, 3),     #28 ->26
            conv2 = L.Convolution2D(10, 10, 4),    #13 ->10
            fc1 = L.Linear(250,10)
        )
        
    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, 2)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, 2)
        y = self.fc1(h)
        return y
       
        
        
if __name__ == '__main__':
    #ハイパーパラメータ
    num_epochs = 10        #エポック数
    batch_size = 500        #バッチ数
    learing_rate = 0.001   #学習率
    
    #データ読み込み
    train, test = load_Mnist()
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    #モデル、オプティマイザ（chainer関数の使用）
    model = ConvNet()
    optimizer = optimizers.Adam(learing_rate)
    optimizer.setup(model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    #訓練ループ
    loss_log = []
    for epoch in range(num_epochs):
        for i in range(0, num_train, batch_size):
                 x_batch = xp.asarray(x_train[i:i+batch_size]) # 1->バッチサイズまでのループ
                 ｃ_batch = xp.asarray(c_train[i:i+batch_size])
                 y_batch = model(x_batch)
        
        #損失関数の計算
                 loss = F.softmax_cross_entropy(y_batch, c_batch)
                 model.cleargrads()              #勾配のリセット
                 loss.backward()                 #重みの更新
                 optimizer.update()
        
        
                 accuracy = F.accuracy(y_batch, c_batch)       #認識率
    
                 print(epoch, accuracy.data, loss.data)          #認識率の表示
                 loss_log.append(loss.data)                      #loss.dataにダータを追加
                 
    #グラフの表示
    plt.plot(loss_log)      
    plt.show()



    
    