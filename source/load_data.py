#coding:utf-8
import pickle
import numpy as np
import cifar10

"""
class LoadData( object ):
    def __init__( self, target_data ):
        self.target_data = target_data
        self.load_mnistaaaaaa = mnist()

    def _mnist( self ):
        # mnistデータよみこみ
        with open('mnist.pkl', 'rb') as file:
            dataset_mnist = pickle.load(file)
        train_data = dataset_mnist['train_img'].reshape((60000, 1,  28, 28))
        train_labels = dataset_mnist['train_label']
        train_data = np.asarray( train_data, dtype = np.float32 )
        train_labels = np.asarray( train_labels, dtype = np.int32 )

        train_data /= 255.0
        valid_data, valid_labels = train_data[50000:], train_labels[50000:]
        train_data, train_labels = train_data[:50000], train_labels[:50000]

        #return train_data, train_labels, valid_data, valid_labels
        return 0

    def _cifar():
        dirCIFAR10 = 'cifar-10-batches-py'
        cifar = cifar10.CIFAR10( dirCIFAR10 )
        X, label, t = cifar.loadData( 'L' )
        X /= 255
        xm = np.mean( X, axis = 0 )
        X -= xm
        X = np.asarray( X, dtype = np.float32 )
        label = np.asarray( label, dtype = np.int32 )
        valid_data, valid_labels = X[40000:], label[40000:]
        train_data, train_labels = X[:40000], label[:40000]

        return train_data, train_labels, valid_data, valid_labels
"""

def load( target_data ):

    if   target_data == "Mnist" : return mnist()
    elif target_data == "Cifar" : return cifar2()



def mnist2():
    # mnistデータよみこみ
    with open('mnist.pkl', 'rb') as file:
        dataset_mnist = pickle.load(file)
    data = dataset_mnist['train_img'].reshape((60000, 1,  28, 28))
    labels = dataset_mnist['train_label']

    # 配列自体をシャッフルすることができないから（データとラベルを同様にシャッフルしなければならないため）
    # 配列のインデックスをシャッフルする
    index = np.array([ i for i in range(60000) ])
    np.random.shuffle(index)

    # カテゴリごとのサンプル数をカウント
    train_num = np.zeros(10, dtype=int)
    valid_num = np.zeros(10, dtype=int)

    # データ、ラベルの配列の雛形
    valid_data = np.zeros(784, dtype=int).reshape((1,28,28))
    train_data = np.zeros(784, dtype=int).reshape((1,28,28))
    valid_labels = np.zeros(1, dtype=int)
    train_labels = np.zeros(1, dtype=int)

    # カテゴリごとのサンプル数
    sample_num = np.array([4000,4000,20,4000,4000,4000,4000,4000,4000,4000])

    for i in range(60000):
        # 検証データが1000未満であるとき
        if valid_num[labels[index[i]]] < 1000:
            valid_data = np.concatenate([valid_data, data[index[i]]], axis=0)
            valid_labels= np.hstack((valid_labels, labels[index[i]]))
            valid_num[labels[index[i]]] += 1
        # 検証データが1000を超えてかつ訓練データが指定数未満であるとき
        elif train_num[labels[index[i]]] < sample_num[labels[index[i]]]:
            train_data = np.concatenate([train_data, data[index[i]]], axis=0)
            train_labels= np.hstack((train_labels, labels[index[i]]))
            train_num[labels[index[i]]] += 1

    # 配列の雛形の削除およびデータの形成
    valid_data = np.delete(valid_data,[0,0], 0).reshape((-1, 1,  28, 28))
    train_data = np.delete(train_data,[0,0], 0).reshape((-1, 1,  28, 28))
    valid_labels = np.delete(valid_labels, 0)
    train_labels = np.delete(train_labels, 0)
    valid_data = np.asarray( valid_data, dtype = np.float32 ) / 255.0
    train_data = np.asarray( train_data, dtype = np.float32 ) / 255.0
    valid_labels = np.asarray( valid_labels, dtype = np.int32 )
    train_labels = np.asarray( train_labels, dtype = np.int32 )

    # カテゴリごとのサンプル数
    num_train_label = [0] * 10
    for i in range(10):
        num_train_label[i] = len(np.where(train_labels==i)[0])

    return train_data, train_labels, valid_data, valid_labels, num_train_label

def cifar2():
    dirCIFAR10 = 'cifar-10-batches-py'
    cifar = cifar10.CIFAR10( dirCIFAR10 )
    data, labels, t = cifar.loadData( 'L' )
    #X /= 255
    tmp = np.mean( data, axis = 0 )
    data -= tmp

    # 配列自体をシャッフルすることができないから（データとラベルを同様にシャッフルしなければならないため）
    # 配列のインデックスをシャッフルする
    index = np.array([ i for i in range(50000) ])
    np.random.shuffle(index)

    # カテゴリごとのサンプル数をカウント
    train_num = np.zeros(10, dtype=int)
    valid_num = np.zeros(10, dtype=int)

    # データ、ラベルの配列の雛形
    valid_data = np.zeros(3072, dtype=int).reshape((3,32,32))
    train_data = np.zeros(3072, dtype=int).reshape((3,32,32))
    valid_labels = np.zeros(1, dtype=int)
    train_labels = np.zeros(1, dtype=int)

    # カテゴリごとのサンプル数
    sample_num = np.array([4000,4000,20,4000,4000,4000,4000,4000,4000,4000])

    for i in range(50000):
        # 検証データが1000未満であるとき
        if valid_num[labels[index[i]]] < 1000:
            valid_data = np.concatenate([valid_data, data[index[i]]], axis=0)
            valid_labels= np.hstack((valid_labels, labels[index[i]]))
            valid_num[labels[index[i]]] += 1
        # 検証データが1000を超えてかつ訓練データが指定数未満であるとき
        elif train_num[labels[index[i]]] < sample_num[labels[index[i]]]:
            train_data = np.concatenate([train_data, data[index[i]]], axis=0)
            train_labels= np.hstack((train_labels, labels[index[i]]))
            train_num[labels[index[i]]] += 1

    # 配列の雛形の削除およびデータの形成
    valid_data = np.delete(valid_data,[0,0], 0).reshape((-1, 3,  32, 32))
    train_data = np.delete(train_data,[0,0], 0).reshape((-1, 3,  32, 32))
    valid_labels = np.delete(valid_labels, 0)
    train_labels = np.delete(train_labels, 0)
    valid_data = np.asarray( valid_data, dtype = np.float32 ) / 255.0
    train_data = np.asarray( train_data, dtype = np.float32 ) / 255.0
    valid_labels = np.asarray( valid_labels, dtype = np.int32 )
    train_labels = np.asarray( train_labels, dtype = np.int32 )

    # カテゴリごとのサンプル数
    num_train_label = [0] * 10
    for i in range(10):
        num_train_label[i] = len(np.where(train_labels==i)[0])

    return train_data, train_labels, valid_data, valid_labels, num_train_label


def mnist():
    # mnistデータよみこみ
    with open('mnist.pkl', 'rb') as file:
        dataset_mnist = pickle.load(file)
    train_data = dataset_mnist['train_img'].reshape((60000, 1,  28, 28))
    train_labels = dataset_mnist['train_label']
    train_data = np.asarray( train_data, dtype = np.float32 )
    train_labels = np.asarray( train_labels, dtype = np.int32 )
    train_data /= 255.0
    valid_data, valid_labels = train_data[50000:], train_labels[50000:]
    train_data, train_labels = train_data[:50000], train_labels[:50000]

    # カテゴリごとのサンプル数
    num_train_label = [0] * 10
    for i in range(10):
        num_train_label[i] = len(np.where(train_labels==i)[0])

    return train_data, train_labels, valid_data, valid_labels, num_train_label
    #return train_data, train_labels, num_train_label


def cifar():
    dirCIFAR10 = 'cifar-10-batches-py'
    cifar = cifar10.CIFAR10( dirCIFAR10 )
    X, label, t = cifar.loadData( 'L' )
    X /= 255
    xm = np.mean( X, axis = 0 )
    X -= xm
    X = np.asarray( X, dtype = np.float32 )
    label = np.asarray( label, dtype = np.int32 )
    valid_data, valid_labels = X[40000:], label[40000:]
    train_data, train_labels = X[:40000], label[:40000]

    # カテゴリごとのサンプル数
    num_train_label = [0] * 10
    for i in range(10):
        num_train_label[i] = len(np.where(train_labels==i)[0])

    return train_data, train_labels, valid_data, valid_labels, num_train_label
