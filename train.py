# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:26:42 2017

@author: kawalab
"""

from copy import deepcopy   # 深い複製
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import configparser
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from mnist_loader import mnist_loader
from mnist_network import CNN


def load_mnist(ndim=2):
    cp = configparser.ConfigParser()
    cp.read('config')
    root_dir = cp.get('dataset_dir', 'dir_path')
    x_train, x_test, c_train, c_test = mnist_loader(ndim, root_dir)
    num_train = len(x_train)
    num_test = len(x_test)
    return x_train, x_test, c_train, c_test, num_train, num_test

def training_parameters():
    cp = configparser.ConfigParser()
    cp.read('config')
    use_device = int(cp.get('Hyper_parameteters', 'gpu_on'))
    num_epochs = int(cp.get('Hyper_parameteters', 'number_epochs'))
    batch_size = int(cp.get('Hyper_parameteters', 'batch_size'))
    learning_rate = float(cp.get('Hyper_parameteters', 'learning_rate'))
    return use_device, num_epochs, batch_size, learning_rate

def train_part(model, num_train, x_train, c_train, xp, batch_size):
    epoch_losses = []               # エポック内の損失値
    epoch_accs = []                 # エポック内の認識率
    for i in tqdm(range(0, num_train, batch_size)):
        x_batch = xp.asarray(x_train[i:i+batch_size], dtype=xp.float32)  # 1->バッチサイズまでのループ
        ｃ_batch = xp.asarray(c_train[i:i+batch_size], dtype=xp.int32)
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
    return train_loss_log, train_acc_log

def validation(model, num_test, x_test, c_test, xp, batch_size):      # バリデーション
    losses = []
    accs = []
    for i in tqdm(range(0, num_test, batch_size)):
        losses = []               # エポック内の損失値
        accs = []                 # エポック内の認識率
        x_batch = xp.asarray(x_test[i:i+batch_size], dtype=xp.float32)  # 1->バッチサイズまでのループ
        ｃ_batch = xp.asarray(c_test[i:i+batch_size], dtype=xp.int32)
        x_batch = chainer.Variable(x_batch)
        ｃ_batch = chainer.Variable(c_batch)
        with chainer.no_backprop_mode():
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
    return loss, test_loss_log, test_acc_log

def save_best_model(loss, best_val_loss, model, best_model, best_epoch):
    # 最小損失ならそのモデルを保持
    if loss.data < best_val_loss:
        best_model = deepcopy(model)
        best_val_loss = loss.data
        best_epoch = epoch
    
    else:
        best_model = best_model
        best_val_loss = best_val_loss
        best_epoch = best_epoch
        
    return best_model, best_val_loss, best_epoch

def print_result_log(epoch, train_loss_log, test_loss_log,
                     train_acc_log, test_acc_log):

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

if __name__ == '__main__':
    (x_train, x_test, c_train, c_test,
     num_train, num_test) = load_mnist(ndim=3)
    gpu, num_epochs, batch_size, learning_rate = training_parameters()
    xp = cuda.cupy if gpu >= 0 else np
    model = CNN()
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
    train_loss_log = []     # 訓練損失関数log
    train_acc_log = []      # 訓練認識率log
    test_loss_log = []      # テスト用損失関数log
    test_acc_log = []       # テスト用認識率log
    best_val_loss = np.inf  # 損失関数最小値保持値
    best_model = []
    best_epoch = []
    for epoch in range(num_epochs):
        train_loss_log, train_acc_log = train_part(model, num_train,
                                                   x_train, c_train,
                                                   xp, batch_size)
        loss, test_loss_log, test_acc_log = validation(model, num_test,
                                                       x_test, c_test,
                                                       xp, batch_size)
        best_model, best_val_loss, best_epoch = save_best_model(loss,
                                                                best_val_loss, 
                                                                model, best_model, best_epoch)

        print_result_log(epoch, train_loss_log, test_loss_log,
                         train_acc_log, test_acc_log)
        
        
        # ハイパーパラメータ等の表示
    print('Hyper Parameters')
    print('min loss = {}'. format(best_val_loss))
    print('Epocks = {}'. format(num_epochs))
    print('batch size = {}'. format(batch_size))
    print('learnig rate = {}'. format(learning_rate))