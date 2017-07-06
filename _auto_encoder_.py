# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 06:05:39 2017

@author: KAWALAB
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


def load_mnist(ndim):
    train, test = get_mnist(ndim=ndim)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


class AutoEncoder1d(chainer.Chain):
    def __init__(self):
        super(AutoEncoder1d, self).__init__(
            l1=L.Linear(784, 500),
            l2=L.Linear(500, 400),
            l3=L.Linear(400, 300),
            l4=L.Linear(300, 400),
            l5=L.Linear(400, 500),
            l6=L.Linear(500, 784)
            )

    def __call__(self, x):
        assert x.ndim == 2        # x.ndimは2である
        assert x.shape[1] == 784  # x.shapeは(??, 784)である
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = F.relu(self.l5(h))
        y = self.l6(h)
        assert y.shape == x.shape  # 入力と出力のshapeが同じである
        return y


if __name__ == '__main__':
    # ハイパーパラメータ
    gpu = 0                # GPU>=0, CPU < 0
    num_epochs = 500    # エポック数
    batch_size = 500        # バッチ数
    learing_rate = 0.001   # 学習率

    xp = cuda.cupy if gpu >= 0 else np

    # データ読み込み
    train, test = load_mnist(1)
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    # モデル、オプティマイザ（chainer関数の使用）
    model = AutoEncoder1d()
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
        for i in tqdm(range(0, num_train, batch_size)):
            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ
            y_batch = model(x_batch)

            # 損失関数の計算
            loss = F.mean_squared_error(y_batch, x_batch)
            model.cleargrads()              # 勾配のリセット
            loss.backward()                 # 重みの更新
            optimizer.update()
            epoch_losses.append(loss.data)

        epoch_loss = np.mean(cuda.to_cpu(xp.stack(epoch_losses)))   # エポックの平均損失
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

    # 答え合わせ
    n = 4   # 確認枚数
    x_batch = xp.asarray(x_test[:n])
    y_batch = best_model(x_batch)
    y_batch = cuda.to_cpu(y_batch.data)
    for i in range(n):
        x = x_test[i]
        # 入力画像
        plt.matshow(x.reshape(28, 28), cmap=plt.cm.gray)
        plt.show()
        # 出力画像
        plt.matshow(y_batch[i].reshape(28, 28),
                    cmap=plt.cm.gray)
        plt.show()