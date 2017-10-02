#coding:utf-8
import numpy as np
import scipy as sp
import datetime
import sys
import pickle
import pandas as pd
from tqdm import tqdm

import cifar10

import layers as convnet
import neural_network as network
import load_data

np.random.seed( 0 )


def train():
    with open("parameter.pkl", 'rb') as file:
        params_dict = pickle.load(file)

    del params_dict['NetworkList'][0]
    print(params_dict['NetworkList'])

    train_data, train_labels, valid_data, valid_labels, num_train_label = load_data.load( params_dict['InputData'] )
    cnn = network.CPRS( params_dict['NetworkList'] )

    #for i in range(10):
        #print("Category(train)" + str(i) + ":" + str(num_train_label[i]))
    #print("\nnum_train_total:" + str(sum(num_train_label)))

    #print(num_train_label[train_labels[0]])
    #print(num_train_label[train_labels[1]])
    #print(num_train_label[train_labels[2]])
    #print(type(num_train_label))


    ##### training
    #
    nepoch = 100000
    batchsize = 100
    eta, mu, lam = 0.01, 0.9, 0.0001


    for i in range(nepoch):

        cnt = 0
        valid_cnt = 0
        if i % int(nepoch / 10) == 0:

            correct_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
            num_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
            acc_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)

            sys.stdout.write("\n")
            for j in range( len(train_data) / batchsize ):
                train_data_batch   = train_data[j*batchsize:(j+1)*batchsize]
                train_labels_batch = train_labels[j*batchsize:(j+1)*batchsize]
                Z = cnn.predict( train_data_batch )
                ZZ = np.argmax( Z, axis=1 )
                cnt += np.sum( train_labels_batch == ZZ )

                for k in range(batchsize):
                    if( train_labels_batch[k] == ZZ[k] ):
                        correct_category[train_labels_batch[k]] += 1
                    num_category[train_labels_batch[k]] += 1

            train_acc_category = correct_category * 100.0 / num_category
            train_acc_balanced = np.sum(train_acc_category) / params_dict["Output"+str(params_dict["LayerNum"]-1)]

            correct_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
            valid_num_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
            acc_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)

            for j in range( len(valid_data) / batchsize ):
                valid_data_batch   = valid_data[j*batchsize:(j+1)*batchsize]
                valid_labels_batch = valid_labels[j*batchsize:(j+1)*batchsize]
                valid_Z = cnn.predict( valid_data_batch )
                ZZ = np.argmax( valid_Z, axis=1 )
                valid_cnt += np.sum( valid_labels_batch == ZZ )
                #print(valid_cnt)]

                for k in range(batchsize):
                    if( valid_labels_batch[k] == ZZ[k] ):
                        correct_category[valid_labels_batch[k]] += 1
                    valid_num_category[valid_labels_batch[k]] += 1

            valid_acc_category = correct_category * 100.0 / valid_num_category
            valid_acc_balanced = np.sum(valid_acc_category) / params_dict["Output"+str(params_dict["LayerNum"]-1)]
            print("+-----------------------------------------+")
            print("|         |     Train     |     Valid     |")
            print("|=========|===============|===============|")
            print("|         |Acc    |Num    |Acc    |Num    |")
            print("|=========================================|")
            for i in range(params_dict["Output"+str(params_dict["LayerNum"]-1)]):
                print("|Class%d   |%04.2f  |%s  |%04.2f  |%s  |" % ( i, train_acc_category[i], "{0:5d}".format(num_category[i]), valid_acc_category[i], "{0:5d}".format(valid_num_category[i])) )

            print("+-----------------------------------------+")
            print("Balanced Train : " + str(train_acc_balanced))
            print("Balanced Valid : " + str(valid_acc_balanced))
            print("         Train : " + str( float(cnt) *100 / len(train_data) ) )
            print("         Valid : " + str( float(valid_cnt) *100 / len(valid_data) ) )
            sys.stdout.write("\n")

        choice = np.random.choice(len(train_data), batchsize)
        XL_batch = train_data[choice]
        labelL_batch = train_labels[choice]

        num_batch = []
        for j in range(batchsize):
            num_batch.append( num_train_label[ labelL_batch[j] ] )
        num_batch = np.array(num_batch)
        num_batch = np.asarray( num_batch, dtype = np.int32 )


        #cnn.train( XL_batch, labelL_batch, eta, mu, lam )
        cnn.train( XL_batch, labelL_batch, eta, mu, lam, num_batch )


        Z = cnn.output( XL_batch )

        #LL  = np.sum( cnn.cost( Z, labelL_batch ) )
        LL  = np.sum( cnn.cost( Z, labelL_batch, num_batch ) )

        sys.stdout.write("\r%d : " % int(i+1))
        sys.stdout.write("%f" % ( LL / batchsize ))
        sys.stdout.flush()



    sys.stdout.write("\n")

    # 重み取得
    for i in range( params_dict['LayerNum'] ):
        if cnn.Layers[i].weight:
            #print(cnn.Layers[i].W.get_value().shape)
            pass
        if cnn.Layers[i].bias:
            #print(cnn.Layers[i].b.get_value().shape)
            pass


    correct_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    num_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    acc_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)

    sys.stdout.write("\n")
    for j in range( len(train_data) / batchsize ):
        train_data_batch   = train_data[j*batchsize:(j+1)*batchsize]
        train_labels_batch = train_labels[j*batchsize:(j+1)*batchsize]
        Z = cnn.predict( train_data_batch )
        ZZ = np.argmax( Z, axis=1 )
        cnt += np.sum( train_labels_batch == ZZ )

        for k in range(batchsize):
            if( train_labels_batch[k] == ZZ[k] ):
                correct_category[train_labels_batch[k]] += 1
            num_category[train_labels_batch[k]] += 1

    train_acc_category = correct_category * 100.0 / num_category
    train_acc_balanced = np.sum(train_acc_category) / params_dict["Output"+str(params_dict["LayerNum"]-1)]


    correct_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    num_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    acc_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)

    sys.stdout.write("\n")
    for j in range( len(valid_data) / batchsize ):
        valid_data_batch   = valid_data[j*batchsize:(j+1)*batchsize]
        valid_labels_batch = valid_labels[j*batchsize:(j+1)*batchsize]
        valid_Z = cnn.predict( valid_data_batch )
        ZZ = np.argmax( valid_Z, axis=1 )
        valid_cnt += np.sum( valid_labels_batch == ZZ )
        #print(valid_cnt)]

        for k in range(batchsize):
            if( valid_labels_batch[k] == ZZ[k] ):
                correct_category[valid_labels_batch[k]] += 1
            num_category[valid_labels_batch[k]] += 1


    valid_acc_category = correct_category * 100.0 / num_category
    valid_acc_balanced = np.sum(valid_acc_category) / params_dict["Output"+str(params_dict["LayerNum"]-1)]
    print("         Train  Valid")
    for i in range(params_dict["Output"+str(params_dict["LayerNum"]-1)]):
        print("Class%d : %2.2f   %2.2f" % ( i, train_acc_category[i], valid_acc_category[i], ) )

    print("Balanced Train : " + str(train_acc_balanced))
    print("Balanced Valid : " + str(valid_acc_balanced))
    print("Train : " + str( float(cnt) *100 / len(train_data) ) )
    print("Valid : " + str( float(valid_cnt) *100 / len(valid_data) ) )
    sys.stdout.write("\n")


if __name__ == "__main__":
    train()
