# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:24:28 2017

@author: kawalab
"""

from copy import deepcopy   # 深い複製
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

def load_mnist():
    train, test = get_mnist(ndim=3)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test

class ConvNet(chainer.Chain):
    def __init__(self):
        super(ConvNet, self).__init__(
            conv1=L.Convolution2D(1, 100, 3),     # 28 ->26
            conv2=L.Convolution2D(100, 100, 4),    # 13 ->10
            fc1=L.Linear(2500, 10)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, 2)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, 2)
        y = self.fc1(h)
        return y

class CNN(chainer.Chain):
    def __init__(self, channel=1, c1=16, c2=32, c3=64, f1=256,
                 f2=512, filter_size1=3, filter_size2=3, filter_size3=3):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(channel, c1, filter_size1),
            conv2=L.Convolution2D(c1, c2, filter_size2),
            conv3=L.Convolution2D(c2, c3, filter_size3),
            l1=L.Linear(f1, f2),
            l2=L.Linear(f2, 10)
            )

    def __call__(self, x):
        # x.data = x.data.reshape((len(x.data), 1, 28, 28))
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.l1(h)))
        y = self.l2(h)
        return y

if __name__ == '__main__':
    # ハイパーパラメータ
    gpu = 0                # GPU>=0, CPU < 0
    num_epochs = 10        # エポック数
    batch_size = 500        # バッチ数
    learing_rate = 0.001   # 学習率

    xp = cuda.cupy if gpu >= 0 else np
   # データ読み込み
    train, test = load_mnist()
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    # モデル、オプティマイザ（chainer関数の使用）
    model = ConvNet()
    optimizer = optimizers.Adam(learing_rate)
    optimizer.setup(model)

    # GPU変換
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 訓練ループ

    # log定義
    train_loss_log = []     # 訓練損失関数log
    train_acc_log = []      # 訓練認識率log
    test_loss_log = []      # テスト用損失関数log
    test_acc_log = []       # テスト用認識率log
    best_val_loss = np.inf  # 損失関数最小値保持値

    # 訓練定義
    for epoch in range(num_epochs):
        epoch_losses = []               # エポック内の損失値
        epoch_accs = []                 # エポック内の認識率
        for i in tqdm(range(0, num_train, batch_size)):
            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ
            ｃ_batch = xp.asarray(c_train[i:i+batch_size])
            y_batch = model(x_batch)

            # 損失関数の計算
            loss = F.softmax_cross_entropy(y_batch, c_batch)
            model.cleargrads()              # 勾配のリセット
            loss.backward()                 # 重みの更新
            accuracy = F.accuracy(y_batch, c_batch)       # 認識率
            optimizer.update()

            epoch_losses.append(loss.data)
            epoch_accs.append(accuracy.data)

        epoch_loss = np.mean(cuda.to_cpu(xp.stack(epoch_losses)))   # エポックの平均損失
        epoch_acc = np.mean(cuda.to_cpu(xp.stack(epoch_accs)))     # エポックの平均認識率
        train_loss_log.append(epoch_loss)
        train_acc_log.append(epoch_acc)

        # バリデーション
        losses = []
        accs = []
        for i in tqdm(range(0, num_train, batch_size)):
            epoch_losses = []               # エポック内の損失値
            epoch_accs = []                 # エポック内の認識率
            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ
            ｃ_batch = xp.asarray(c_train[i:i+batch_size])

            x_batch = chainer.Variable(x_batch, volatile=True)
            
            
            ｃ_batch = chainer.Variable(c_batch, volatile=True)
            
            
            y_batch = model(x_batch)

            # 損失関数の計算
            loss = F.softmax_cross_entropy(y_batch, c_batch)
            accuracy = F.accuracy(y_batch, c_batch)       # 認識率

            losses.append(loss.data)
            accs.append(accuracy.data)
        test_loss = np.mean(cuda.to_cpu(xp.stack(losses)))   # エポックの平均損失
        test_acc = np.mean(cuda.to_cpu(xp.stack(accs)))     # エポックの平均認識率
        test_loss_log.append(test_loss)
        test_acc_log.append(test_acc)

        # 最小損失ならそのモデルを保持
        if loss.data < best_val_loss:
            best_model = deepcopy(model)
            best_val_loss = loss.data
            best_epoch = epoch

        # エポック数、認識率、損失値の表示
        print('{}: acc={}, loss={}'.format(
                epoch, epoch_acc, epoch_loss))

    # グラフの表示
        plt.figure(figsize=(10, 4))
        plt.title('Loss')
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_log, label='train loss')
        plt.plot(test_loss_log, label='test loss')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.plot(train_acc_log, label='train acc')
        plt.plot(test_acc_log, label='test acc')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
'''
    # CNNによる訓練
    model = CNN()
    optimizer = optimizers.Adam(learing_rate)
    optimizer.setup(model)

    if gpu >= 0:
        cuda.get_device(gpu).use()

        model.to_gpu()



    # 訓練ループ

    loss_log2 = []

    for epoch in range(num_epochs):

        for i in range(0, num_train, batch_size):

            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ

            ｃ_batch = xp.asarray(c_train[i:i+batch_size])

            y_batch = model(x_batch)



            # 損失関数の計算

            loss = F.softmax_cross_entropy(y_batch, c_batch)

            model.cleargrads()              # 勾配のリセット

            loss.backward()                 # 重みの更新

            optimizer.update()



            accuracy = F.accuracy(y_batch, c_batch)       # 認識率



            print(epoch, accuracy.data, loss.data)          # 認識率の表示

            loss_log2.append(cuda.to_cpu(loss.data))         # loss.logにデータを追加



    # グラフの表示

    plt.plot(loss_log1)

    plt.show()



    # グラフの表示

    plt.plot(loss_log2)

    plt.show()

'''