# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:54:00 2017
@author: kawalab
"""

from copy import deepcopy   # 深い複製

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# from mnist_loader import load_mnist

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer.datasets import get_mnist
from chainer.dataset import concat_examples

def load_mnist(ndim):
    train, test = get_mnist(ndim=ndim)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


class AutoEncoder2d(chainer.Chain):
    def __init__(self):
        super(AutoEncoder2d, self).__init__(
            conv1=L.Convolution2D(1, 32, 3, pad=1),
            conv2=L.Convolution2D(32, 64, 3),
            conv3=L.Convolution2D(64, 128, 3),

            dconv3=L.Deconvolution2D(128, 64, 3),
            dconv2=L.Deconvolution2D(64, 32, 3),
            dconv1=L.Deconvolution2D(32, 1, 3, pad=1)
            )

    def __call__(self, x):
        outsize1 = x.shape[-2:]
        h = F.relu(self.conv1(x))           # 28
        h = F.max_pooling_2d(h, 2)          # 14
        h = F.relu(self.conv2(h))           # 12
        outsize2 = h.shape[-2:]
        h = F.max_pooling_2d(h, 2)          # 6
        h = F.relu(self.conv3(h))           # 4

        h = F.relu(self.dconv3(h))          # 6
        h = F.unpooling_2d(h, 2, outsize=outsize2)# 12
        h = F.relu(self.dconv2(h))          # 14
        h = F.unpooling_2d(h, 2, outsize=outsize1)# 28
        y = F.sigmoid(self.dconv1(h))       # 28
        return y

if __name__ == '__main__':
    # ハイパーパラメータ
    gpu = 0                # GPU>=0, CPU < 0
    num_epochs = 200    # エポック数
    batch_size = 200        # バッチ数
    learing_rate = 0.0001   # 学習率

    xp = cuda.cupy if gpu >= 0 else np #GPUだったらcupy，CPUだったらnumpy

    # データ読み込み
    train, test = load_mnist(3)
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    # モデル、オプティマイザ（chainer関数の使用）
    model = AutoEncoder2d()
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
    try:
        for epoch in range(num_epochs):
            epoch_losses = []               # エポック内の損失値
            for i in tqdm(range(0, num_train, batch_size)):
                x_batch = xp.asarray(x_train[i:i+batch_size])
                y_batch = model(x_batch)

                # 損失関数の計算
                loss = F.mean_squared_error(y_batch, x_batch)
                model.cleargrads()              # 勾配のリセット
                loss.backward()                 # 重みの更新
                optimizer.update()
                epoch_losses.append(loss.data)

            epoch_loss = np.mean(cuda.to_cpu(xp.stack(epoch_losses)))
            train_loss_log.append(epoch_loss)

            # バリデーション
            losses = []
            accs = []
            for i in tqdm(range(0, num_test, batch_size)):
                epoch_losses = []              # エポック内の損失値
                x_batch = xp.asarray(x_test[i:i+batch_size])  # 1->バッチサイズまでのループ
                x_batch = chainer.Variable(x_batch, volatile=True)
                y_batch = model(x_batch)

                # 損失関数の計算
                loss = F.mean_squared_error(y_batch, x_batch)
                losses.append(loss.data)
            test_loss = np.mean(cuda.to_cpu(xp.stack(losses)))   # エポックの平均損失
            test_loss_log.append(test_loss)

            # 最小損失ならそのモデルを保持
            if loss.data < best_val_loss:
                best_model = deepcopy(model)
                best_val_loss = loss.data
                best_epoch = epoch

            # エポック数、認識率、損失値の表示
            print('{}: loss = {}'.format(epoch, epoch_loss))

            # グラフの表示
            plt.figure(figsize=(10, 4))
            plt.title('Loss')
            plt.plot(train_loss_log, label='train loss')
            plt.plot(test_loss_log, label='test loss')
            plt.legend()
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print('Interrupted by Ctrl+c!')
    # 答え合わせ
    n = 4   # 確認枚数
    x_batch = xp.asarray(x_test[:n])
    y_batch = best_model(x_batch)
    y_batch = cuda.to_cpu(y_batch.data)
    for i in range(n):
        # 入力画像
        plt.matshow(cuda.to_cpu(x_batch[i][0]), cmap=plt.cm.gray)
        plt.show()
        # 出力画像
        plt.matshow(y_batch[i][0], cmap=plt.cm.gray)
        plt.show()

    # ハイパーパラメータ等の表示
    print('Hyper Parameters')
    print('min loss = {}'. format(best_val_loss))
    print('Epocks = {}'. format(num_epochs))
    print('batch size = {}'. format(batch_size))
    print('learnig rate = {}'. format(learing_rate))