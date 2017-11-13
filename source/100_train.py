#coding:utf-8
import numpy as np
#import scipy as sp
#import datetime
import sys
import pickle
import pandas as pd
#from tqdm import tqdm


#import cifar10

#import layers as convnet
import neural_network as network
import load_data

def predict( lap, seed, params_dict, cnn, train_data, train_labels, valid_data, valid_labels, batchsize, cnt, valid_cnt ):
    correct_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    num_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    acc_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)

    sys.stdout.write("\n")
    for i in range( len(train_data) / batchsize ):
        train_data_batch   = train_data[i*batchsize:(i+1)*batchsize]
        train_labels_batch = train_labels[i*batchsize:(i+1)*batchsize]
        Z = cnn.predict( train_data_batch )
        ZZ = np.argmax( Z, axis=1 )
        cnt += np.sum( train_labels_batch == ZZ )

        for j in range(batchsize):
            if( train_labels_batch[j] == ZZ[j] ):
                correct_category[train_labels_batch[j]] += 1
            num_category[train_labels_batch[j]] += 1

    train_acc_category = correct_category * 100.0 / num_category
    train_acc_balanced = np.sum(train_acc_category) / params_dict["Output"+str(params_dict["LayerNum"]-1)]

    correct_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    valid_num_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)
    acc_category = np.zeros(params_dict["Output"+str(params_dict["LayerNum"]-1)], dtype=int)

    for i in range( len(valid_data) / batchsize ):
        valid_data_batch   = valid_data[i*batchsize:(i+1)*batchsize]
        valid_labels_batch = valid_labels[i*batchsize:(i+1)*batchsize]
        valid_Z = cnn.predict( valid_data_batch )
        ZZ = np.argmax( valid_Z, axis=1 )
        valid_cnt += np.sum( valid_labels_batch == ZZ )
        #print(valid_cnt)]

        for j in range(batchsize):
            if( valid_labels_batch[j] == ZZ[j] ):
                correct_category[valid_labels_batch[j]] += 1
            valid_num_category[valid_labels_batch[j]] += 1

    valid_acc_category = correct_category * 100.0 / valid_num_category
    valid_acc_balanced = np.sum(valid_acc_category) / params_dict["Output"+str(params_dict["LayerNum"]-1)]
    train_acc_total = float(cnt) *100 / len(train_data)
    valid_acc_total = float(valid_cnt) *100 / len(valid_data)
    print("+-----------------------------------------+")
    print("|Lap:" + str(lap) + "  |     Train     |     Valid     |")
    print("|=========|===============|===============|")
    print("|SEED:" + str(seed) + "|Acc    |Num    |Acc    |Num    |")
    print("|=========================================|")
    for i in range(params_dict["Output"+str(params_dict["LayerNum"]-1)]):
        print("|Class%d   |%04.2f  |%s  |%04.2f  |%s  |" % ( i, train_acc_category[i], "{0:5d}".format(num_category[i]), valid_acc_category[i], "{0:5d}".format(valid_num_category[i])) )

    print("+-----------------------------------------+")
    print("Balanced Train : " + str(train_acc_balanced))
    print("Balanced Valid : " + str(valid_acc_balanced))
    print("         Train : " + str( train_acc_total ) )
    print("         Valid : " + str( valid_acc_total ) )
    sys.stdout.write("\n")

    return train_acc_category, valid_acc_category, train_acc_balanced, valid_acc_balanced, train_acc_total, valid_acc_total



def train():
    # パラメータファイル読み取り
    with open("parameter.pkl", 'rb') as file:
        params_dict = pickle.load(file)
    del params_dict['NetworkList'][0]
    print(params_dict['NetworkList'])
    print("Data Loading...")

    # 画像データ読み込み
    train_data, train_labels, valid_data, valid_labels, num_train_label = load_data.load( params_dict['InputData'] )
    # ニューラルネットワーク構築
    cnn = network.CPRS( params_dict['NetworkList'] )

    # csvファイル出力のための配列
    #result = np.empty((0,1), int)
    df = pd.DataFrame([[num_train_label[0], 1000],[num_train_label[1], 1000], \
                       [num_train_label[2], 1000],[num_train_label[3], 1000], \
                       [num_train_label[4], 1000],[num_train_label[5], 1000], \
                       [num_train_label[6], 1000],[num_train_label[7], 1000], \
                       [num_train_label[8], 1000],[num_train_label[9], 1000], \
                       [sum(num_train_label), 10000], [sum(num_train_label), 10000]]).T

    df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
    df.index   = ["Train:Num","Valid:Num"]
    #print(df)
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    #df = pd.DataFrame(np.random.random([100, 3]), columns=['foo', 'bar', 'baz'])

    # 100回繰り返す
    loop = 100
    result_train_all = np.empty((loop, 12), float)
    result_valid_all = np.empty((loop, 12), float)
    print("Start Training...")
    for i in range(loop):
        # SEED値を設定し、乱数更新
        seed = np.random.randint(10000)
        np.random.seed(seed)
        #print np.random.rand() # ここは一致

        #result = np.append(result, np.array([[seed]]), axis=0)

        ######## training #########
        nepoch = 10000
        batchsize = 100
        eta, mu, lam = 0.01, 0.9, 0.0001

        result_train_acc     = np.empty((0,10), float)
        result_valid_acc     = np.empty((0,10), float)
        result_train_total   = np.empty((0,1), float)
        result_valid_total   = np.empty((0,1), float)
        result_train_balance = np.empty((0,1), float)
        result_valid_balance = np.empty((0,1), float)
        for j in range(nepoch):

            cnt = 0
            valid_cnt = 0
            if j % int(nepoch / 20) == 0:
                train_acc_category, valid_acc_category, train_acc_balanced, valid_acc_balanced, train_acc_total, valid_acc_total = \
                    predict( i+1, seed, params_dict, cnn, train_data, train_labels, valid_data, valid_labels, batchsize, cnt, valid_cnt )

                # 初回だけ繰り返しごとの精度を出力
                if(i==0):
                    add_train_df = pd.DataFrame([train_acc_category[0], train_acc_category[1], \
                                             train_acc_category[2], train_acc_category[3], \
                                             train_acc_category[4], train_acc_category[5], \
                                             train_acc_category[6], train_acc_category[7], \
                                             train_acc_category[8], train_acc_category[9], \
                                             train_acc_total, train_acc_balanced]).T

                    add_train_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
                    add_train_df.index   = ["Train:" + str(j)]
                    #pd.concat([df, add_df], axis=0)
                    train_df = train_df.append(add_train_df)

                    add_valid_df = pd.DataFrame([valid_acc_category[0], valid_acc_category[1], \
                                             valid_acc_category[2], valid_acc_category[3], \
                                             valid_acc_category[4], valid_acc_category[5], \
                                             valid_acc_category[6], valid_acc_category[7], \
                                             valid_acc_category[8], valid_acc_category[9], \
                                             valid_acc_total, valid_acc_balanced]).T

                    add_valid_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
                    add_valid_df.index   = ["valid:" + str(j)]
                    #pd.concat([df, add_df], axis=0)
                    valid_df = valid_df.append(add_valid_df)

                if float(j) / nepoch > 0.5:
                    result_train_acc     = np.append(result_train_acc, np.array([train_acc_category]), axis=0)
                    result_valid_acc     = np.append(result_valid_acc, np.array([train_acc_category]), axis=0)
                    result_train_total   = np.append(result_train_total, np.array([[train_acc_total]]), axis=0)
                    result_valid_total   = np.append(result_valid_total, np.array([[valid_acc_total]]), axis=0)
                    result_train_balance = np.append(result_train_balance, np.array([[train_acc_balanced]]), axis=0)
                    result_valid_balance = np.append(result_valid_balance, np.array([[valid_acc_balanced]]), axis=0)



            # バッチサイズ分の画像データ、ラベルをランダムに選出
            choice = np.random.choice(len(train_data), batchsize, replace=False)
            #print np.random.rand() # ここは違う
            #print choice
            XL_batch = train_data[choice]
            labelL_batch = train_labels[choice]

            # バッチの画像のクラスごとの枚数
            num_batch = []
            for k in range(batchsize):
                num_batch.append( num_train_label[ labelL_batch[k] ] )
            # 配列に変換
            num_batch = np.array(num_batch)
            num_batch = np.asarray( num_batch, dtype = np.int32 )

            # 学習
            #cnn.train( XL_batch, labelL_batch, eta, mu, lam )
            cnn.train( XL_batch, labelL_batch, eta, mu, lam, num_batch )

            # 損失関数の値を計算
            Z = cnn.output( XL_batch )
            #LL  = np.sum( cnn.cost( Z, labelL_batch ) )
            LL  = np.sum( cnn.cost( Z, labelL_batch, num_batch ) )
            # 損失関数の値を出力
            sys.stdout.write("\r%d : " % int(j+1))
            sys.stdout.write("%f" % ( LL / batchsize ))
            sys.stdout.flush()

        sys.stdout.write("\n")

        train_acc_category, valid_acc_category, train_acc_balanced, valid_acc_balanced, train_acc_total, valid_acc_total = \
            predict( i+1, seed, params_dict, cnn, train_data, train_labels, valid_data, valid_labels, batchsize, cnt, valid_cnt )

        # 初回だけ繰り返しごとの精度を出力
        if(i==0):
            add_train_df = pd.DataFrame([train_acc_category[0], train_acc_category[1], \
                                     train_acc_category[2], train_acc_category[3], \
                                     train_acc_category[4], train_acc_category[5], \
                                     train_acc_category[6], train_acc_category[7], \
                                     train_acc_category[8], train_acc_category[9], \
                                     train_acc_total, train_acc_balanced]).T

            add_train_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
            add_train_df.index   = ["Train:" + str(j+1)]
            #pd.concat([df, add_df], axis=0)
            train_df = train_df.append(add_train_df)

            add_valid_df = pd.DataFrame([valid_acc_category[0], valid_acc_category[1], \
                                     valid_acc_category[2], valid_acc_category[3], \
                                     valid_acc_category[4], valid_acc_category[5], \
                                     valid_acc_category[6], valid_acc_category[7], \
                                     valid_acc_category[8], valid_acc_category[9], \
                                     valid_acc_total, valid_acc_balanced]).T

            add_valid_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
            add_valid_df.index   = ["valid:" + str(j+1)]
            #pd.concat([df, add_df], axis=0)
            valid_df = valid_df.append(add_valid_df)


        result_train_acc     = np.append(result_train_acc, np.array([train_acc_category]), axis=0)
        result_valid_acc     = np.append(result_valid_acc, np.array([valid_acc_category]), axis=0)
        result_train_total   = np.append(result_train_total, np.array([[train_acc_total]]), axis=0)
        result_valid_total   = np.append(result_valid_total, np.array([[valid_acc_total]]), axis=0)
        result_train_balance = np.append(result_train_balance, np.array([[train_acc_balanced]]), axis=0)
        result_valid_balance = np.append(result_valid_balance, np.array([[valid_acc_balanced]]), axis=0)

        ave_train_acc     = np.mean(result_train_acc, axis=0)
        ave_valid_acc     = np.mean(result_valid_acc, axis=0)
        ave_train_total   = np.mean(result_train_total)
        ave_valid_total   = np.mean(result_valid_total)
        ave_train_balance = np.mean(result_train_balance)
        ave_valid_balance = np.mean(result_valid_balance)

        for j in range(10):
            result_train_all[i][j]   = ave_train_acc[j]
            result_valid_all[i][j] = ave_valid_acc[j]
        result_train_all[i][10], result_train_all[i][11] = ave_train_total, ave_train_balance
        result_valid_all[i][10], result_train_all[i][11] = ave_valid_total, ave_valid_balance


        add_df = pd.DataFrame([[ave_train_acc[0], ave_valid_acc[0]], \
                               [ave_train_acc[1], ave_valid_acc[1]], \
                               [ave_train_acc[2], ave_valid_acc[2]], \
                               [ave_train_acc[3], ave_valid_acc[3]], \
                               [ave_train_acc[4], ave_valid_acc[4]], \
                               [ave_train_acc[5], ave_valid_acc[5]], \
                               [ave_train_acc[6], ave_valid_acc[6]], \
                               [ave_train_acc[7], ave_valid_acc[7]], \
                               [ave_train_acc[8], ave_valid_acc[8]], \
                               [ave_train_acc[9], ave_valid_acc[9]], \
                               [ave_train_total, ave_valid_total], \
                               [ave_train_balance, ave_valid_balance]]).T

        add_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
        add_df.index   = ["Train:SEED" + str(seed), "Valid:SEED" + str(seed)]
        #pd.concat([df, add_df], axis=0)
        df = df.append(add_df)

    result_all = np.empty((2, 12), float)
    result_all[0] = np.mean(result_train_all, axis=0)
    result_all[1] = np.mean(result_valid_all, axis=0)

    add_df = pd.DataFrame([[result_all[0][0], result_all[1][0]], \
                           [result_all[0][1], result_all[1][1]], \
                           [result_all[0][2], result_all[1][2]], \
                           [result_all[0][3], result_all[1][3]], \
                           [result_all[0][4], result_all[1][4]], \
                           [result_all[0][5], result_all[1][5]], \
                           [result_all[0][6], result_all[1][6]], \
                           [result_all[0][7], result_all[1][7]], \
                           [result_all[0][8], result_all[1][8]], \
                           [result_all[0][9], result_all[1][9]], \
                           [ave_train_total, ave_valid_total], \
                           [ave_train_balance, ave_valid_balance]]).T
    add_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
    add_df.index   = ["Train:All", "Valid:All"]
    #pd.concat([df, add_df], axis=0)
    df = df.append(add_df)
    df = df.append(train_df)
    df = df.append(valid_df)

    # 結果の配列をデータフレームに変換、CSVファイル出力
    #df = pd.DataFrame(result)
    df.to_csv("result" + str(np.random.randint(100000)) + ".csv")



if __name__ == "__main__":
    train()
