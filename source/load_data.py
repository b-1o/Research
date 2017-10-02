#coding:utf-8
import pickle
import os
import numpy as np
import pandas as pd
import cifar10
from PIL import Image

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
    #if   target_data == "mnist" : return mnist()
    #elif target_data == "cifar" : return cifar()

    if   target_data == "Mnist" : return mnist()
    elif target_data == "Cifar" : return cifar()
    elif target_data == "Uniqlo" : return uniqlo()

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


def uniqlo():
    train_data = []
    train_labels = []
    data = pd.read_csv("train_master.tsv", sep='\t')

    for row in data.iterrows():
        file, label = row[1]['file_name'], row[1]['category_id']
        try:
            im = Image.open(os.path.join("..", "data", "train", file))
            #print(file)

            # きれいじゃない・・・
            """
            if label == 1 or label == 2 or label == 10 or label == 11 or label == 13 or label == 14 or label == 15 or label == 16 or label == 17 or label == 19 or label == 20:
                for i in range(4):
                    inflation_image(im, label, file_number, image_size, train_data, train_labels, train_files)
            """
            #print(row)

            # 画像の縮小
            image_size = 50
            im = im.resize((image_size, image_size))

            train_data.append(np.array(im) / 255.0)
            train_labels.append(label)
            #train_files.append(file_number)
            #file_number += 1

            #if  datetime.datetime.now() > load_time + datetime.timedelta(seconds=10):
                #print( "{0:.1f}".format(len(train_data) * 100 / len(data)) + "%")
                #load_time += datetime.timedelta(seconds=10)


        except Exception as e:
            print(str(e))

    # 画像データ、ラベル、ファイル名を配列に格納
    train_data = np.array(train_data).transpose(0, 3, 1, 2)
    train_labels = np.array(train_labels)
    train_data = np.asarray( train_data, dtype = np.float32 )
    train_labels = np.asarray( train_labels, dtype = np.int32 )
    valid_data, valid_labels = train_data[10000:], train_labels[10000:]
    train_data, train_labels = train_data[:10000], train_labels[:10000]
    #valid_data, valid_labels = train_data[1000:], train_labels[1000:]
    #train_data, train_labels = train_data[:1000], train_labels[:1000]
    #train_files = np.array(train_files)

    # カテゴリごとのサンプル数
    num_train_label = [0] * 24
    for i in range(24):
        num_train_label[i] = len(np.where(train_labels==i)[0])


    return train_data, train_labels, valid_data, valid_labels, num_train_label
