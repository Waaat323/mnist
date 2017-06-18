# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:13:26 2017
@author: sakurai
MNIST dataset.
It comsists of 60,000 training examples and 10,000 testing examples.
x: each image is an ndarray of uint8 dtype, 1*28*28 shape.
c: scalar of uint32
"""

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer.datasets import get_mnist
from chainer.dataset import concat_examples


def load_mnist_as_ndarray():
    train, test = get_mnist(ndim=3)
    train = concat_examples(train)
    test = concat_examples(test)
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

if __name__ == '__main__':
    gpu = 0
    num_epochs = 10
    batch_size = 500
    learning_rate = 0.001

    xp = cuda.cupy if gpu >= 0 else np

    # データ読み込み
    train, test = load_mnist_as_ndarray()
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    # モデルとオプティマイザの準備
    model = ConvNet()
    optimaizer = optimizers.Adam(learning_rate)
    optimaizer.setup(model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 訓練ループ
    train_loss_log = []
    train_acc_log = []  # accuracy
    test_loss_log = []
    test_acc_log = []  # accuracy
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_accs = []
        # 訓練
        for i in tqdm(range(0, num_train, batch_size)[:50]):
            x_batch = xp.asarray(x_train[i:i + batch_size])
            c_batch = xp.asarray(c_train[i:i + batch_size])
            y_batch = model(x_batch)

            loss = F.softmax_cross_entropy(y_batch, c_batch)
            model.cleargrads()
            loss.backward()
            accuracy = F.accuracy(y_batch, c_batch)
            optimaizer.update()

            epoch_losses.append(loss.data)
            epoch_accs.append(accuracy.data)

        epoch_loss = np.mean(cuda.to_cpu(xp.array(epoch_losses)))
        epoch_acc = np.mean(cuda.to_cpu(xp.array(epoch_accs)))
        train_loss_log.append(epoch_loss)
        train_acc_log.append(epoch_acc)

        # バリデーション
        losses = []
        accs = []
        for i in tqdm(range(0, num_test, batch_size)[:50]):
            x_batch = xp.asarray(x_test[i:i + batch_size])
            c_batch = xp.asarray(c_test[i:i + batch_size])

            x_batch = chainer.Variable(x_batch, volatile=True)
            c_batch = chainer.Variable(c_batch, volatile=True)
            y_batch = model(x_batch)

            loss = F.softmax_cross_entropy(y_batch, c_batch)
            accuracy = F.accuracy(y_batch, c_batch)

            losses.append(loss.data)
            accs.append(accuracy.data)
        test_loss = np.mean(cuda.to_cpu(xp.array(losses)))
        test_acc = np.mean(cuda.to_cpu(xp.array(accs)))
        test_loss_log.append(test_loss)
        test_acc_log.append(test_acc)

        if loss.data < best_val_loss:
            best_model = deepcopy(model)
            best_val_loss = loss.data
            best_epoch = epoch

        print('{}: acc={}, loss={}'.format(
            epoch, epoch_acc, epoch_loss))

        plt.figure(figsize=(10, 4))
        plt.title('Loss')
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_log, label='train loss')
        plt.plot(test_loss_log, label='train acc')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.plot(train_acc_log)
        plt.plot(test_acc_log)
        plt.ylim([0.0, 1.0])
        plt.legend(['val loss', 'val acc'])
        plt.grid()

        plt.tight_layout()
        plt.show()

    # テストセットの評価
    # best_model(test_x)