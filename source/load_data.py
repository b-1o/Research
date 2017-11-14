#coding:utf-8
import pickle
import numpy as np
import cifar10
from tqdm import tqdm

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

def load( target_data, sample_num ):

    #if   target_data == "Mnist" : return mnist()
    if   target_data == "Mnist" : return mnist3(sample_num)
    elif target_data == "Cifar" : return cifar2()


def mnist3(sample_num):
    # mnistデータよみこみ
    with open('mnist_reshape.pkl', 'rb') as file:
        dataset_mnist = pickle.load(file)
    class0 = dataset_mnist['class0']
    class1 = dataset_mnist['class1']
    class2 = dataset_mnist['class2']
    class3 = dataset_mnist['class3']
    class4 = dataset_mnist['class4']
    class5 = dataset_mnist['class5']
    class6 = dataset_mnist['class6']
    class7 = dataset_mnist['class7']
    class8 = dataset_mnist['class8']
    class9 = dataset_mnist['class9']
    class0_labels = dataset_mnist['class0_labels']
    class1_labels = dataset_mnist['class1_labels']
    class2_labels = dataset_mnist['class2_labels']
    class3_labels = dataset_mnist['class3_labels']
    class4_labels = dataset_mnist['class4_labels']
    class5_labels = dataset_mnist['class5_labels']
    class6_labels = dataset_mnist['class6_labels']
    class7_labels = dataset_mnist['class7_labels']
    class8_labels = dataset_mnist['class8_labels']
    class9_labels = dataset_mnist['class9_labels']

    choice0 = np.random.choice(len(class0), sample_num[0], replace=False)
    choice1 = np.random.choice(len(class1), sample_num[1], replace=False)
    choice2 = np.random.choice(len(class2), sample_num[2], replace=False)
    choice3 = np.random.choice(len(class3), sample_num[3], replace=False)
    choice4 = np.random.choice(len(class4), sample_num[4], replace=False)
    choice5 = np.random.choice(len(class5), sample_num[5], replace=False)
    choice6 = np.random.choice(len(class6), sample_num[6], replace=False)
    choice7 = np.random.choice(len(class7), sample_num[7], replace=False)
    choice8 = np.random.choice(len(class8), sample_num[8], replace=False)
    choice9 = np.random.choice(len(class9), sample_num[9], replace=False)

    data_class0 = class0[choice0]
    data_class1 = class1[choice1]
    data_class2 = class2[choice2]
    data_class3 = class3[choice3]
    data_class4 = class4[choice4]
    data_class5 = class5[choice5]
    data_class6 = class6[choice6]
    data_class7 = class7[choice7]
    data_class8 = class8[choice8]
    data_class9 = class9[choice9]

    data_class0_labels = class0_labels[choice0]
    data_class1_labels = class1_labels[choice1]
    data_class2_labels = class2_labels[choice2]
    data_class3_labels = class3_labels[choice3]
    data_class4_labels = class4_labels[choice4]
    data_class5_labels = class5_labels[choice5]
    data_class6_labels = class6_labels[choice6]
    data_class7_labels = class7_labels[choice7]
    data_class8_labels = class8_labels[choice8]
    data_class9_labels = class9_labels[choice9]

    train_data = data_class0
    train_data = np.append(train_data, data_class1, axis=0)
    train_data = np.append(train_data, data_class2, axis=0)
    train_data = np.append(train_data, data_class3, axis=0)
    train_data = np.append(train_data, data_class4, axis=0)
    train_data = np.append(train_data, data_class5, axis=0)
    train_data = np.append(train_data, data_class6, axis=0)
    train_data = np.append(train_data, data_class7, axis=0)
    train_data = np.append(train_data, data_class8, axis=0)
    train_data = np.append(train_data, data_class9, axis=0)
    train_data = train_data.reshape((-1, 1,  28, 28))

    train_data_labels = data_class0_labels
    train_data_labels = np.append(train_data_labels, data_class1_labels)
    train_data_labels = np.append(train_data_labels, data_class2_labels)
    train_data_labels = np.append(train_data_labels, data_class3_labels)
    train_data_labels = np.append(train_data_labels, data_class4_labels)
    train_data_labels = np.append(train_data_labels, data_class5_labels)
    train_data_labels = np.append(train_data_labels, data_class6_labels)
    train_data_labels = np.append(train_data_labels, data_class7_labels)
    train_data_labels = np.append(train_data_labels, data_class8_labels)
    train_data_labels = np.append(train_data_labels, data_class9_labels)
    train_data_labels = np.asarray( train_data_labels, dtype = np.int32 )



    valid_data = dataset_mnist['test_img'].reshape((-1, 1,  28, 28))
    valid_labels = dataset_mnist['test_label']
    valid_data = np.asarray( valid_data, dtype = np.float32 ) / 255.0

    # カテゴリごとのサンプル数
    num_train_label = [0] * 10
    for i in range(10):
        num_train_label[i] = len(np.where(train_data_labels==i)[0])

    return train_data, train_data_labels, valid_data, valid_labels, num_train_label


def mnist2(sample_num):
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
    #sample_num = np.array([5000,5000,5000,5000,5000,5000,5000,5000,5000,5000])

    for i in tqdm(range(60000)):
        # 検証データが1000未満であるとき
        #if valid_num[labels[index[i]]] < 1000:
        #    valid_data = np.concatenate([valid_data, data[index[i]]], axis=0)
        #    valid_labels= np.hstack((valid_labels, labels[index[i]]))
        #    valid_num[labels[index[i]]] += 1
        # 検証データが1000を超えてかつ訓練データが指定数未満であるとき
        if train_num[labels[index[i]]] < sample_num[labels[index[i]]]:
            train_data = np.concatenate([train_data, data[index[i]]], axis=0)
            train_labels= np.hstack((train_labels, labels[index[i]]))
            train_num[labels[index[i]]] += 1

    valid_data = dataset_mnist['test_img'].reshape((-1, 1,  28, 28))
    valid_labels = dataset_mnist['test_label']

    # 配列の雛形の削除およびデータの形成
    #valid_data = np.delete(valid_data,[0,0], 0).reshape((-1, 1,  28, 28))
    train_data = np.delete(train_data,[0,0], 0).reshape((-1, 1,  28, 28))
    #valid_labels = np.delete(valid_labels, 0)
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
    sample_num = np.array([4000,4000,4000,4000,4000,4000,4000,4000,4000,4000])

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

    valid_data = dataset_mnist['test_img'].reshape((-1, 1,  28, 28))
    valid_labels = dataset_mnist['test_label']
    valid_data = np.asarray( valid_data, dtype = np.float32 ) / 255.0


    print(valid_data[0])
    print(train_data[0])

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
