#coding:utf-8
import numpy as np
import sys
import pickle
import pandas as pd

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


def train(sample_num):
    # パラメータファイル読み取り
    with open("parameter.pkl", 'rb') as file:
        params_dict = pickle.load(file)
    del params_dict['NetworkList'][0]
    print(params_dict['NetworkList'])
    print("Data Loading...")

    # 画像データ読み込み
    train_data, train_labels, valid_data, valid_labels, num_train_label = load_data.load( params_dict['InputData'], sample_num )

    # csvファイル出力のための配列
    #result = np.empty((0,1), int)
    df = pd.DataFrame([[num_train_label[0], 980], [num_train_label[1], 1135], \
                       [num_train_label[2], 1032],[num_train_label[3], 1010], \
                       [num_train_label[4], 982], [num_train_label[5], 892],  \
                       [num_train_label[6], 958], [num_train_label[7], 1028], \
                       [num_train_label[8], 974], [num_train_label[9], 1009], \
                       [sum(num_train_label), 10000], [sum(num_train_label), 10000]]).T

    df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
    df.index   = ["Train:Num","Valid:Num"]
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    all_df   = pd.DataFrame()

    # 100回繰り返す
    # 30回でおｋ
    loop = 30
    result_train_all = np.empty((0, 12), np.float32)
    result_valid_all = np.empty((0, 12), np.float32)
    print("Start Training...")
    for i in range(loop):
        # SEED値を設定し、乱数更新
        seed = np.random.randint(10000)
        np.random.seed(seed)
        #print np.random.rand() # ここは一致

        # ニューラルネットワーク構築
        cnn = network.CPRS( params_dict['NetworkList'] )

        ######## training #########
        nepoch = 20000
        batchsize = 100
        eta, mu, lam = 0.01, 0.9, 0.0001

        result_train = np.empty((0, 12), np.float32)
        result_valid = np.empty((0, 12), np.float32)

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
                    result_train_one = np.append(train_acc_category, train_acc_total)
                    result_train_one = np.append(result_train_one, train_acc_balanced).reshape(1,12)
                    result_train = np.append(result_train, result_train_one, axis=0)
                    result_valid_one = np.append(valid_acc_category, valid_acc_total)
                    result_valid_one = np.append(result_valid_one, valid_acc_balanced).reshape(1,12)
                    result_valid = np.append(result_valid, result_valid_one, axis=0)



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
            valid_df = valid_df.append(add_valid_df)

        if float(j) / nepoch > 0.5:
            result_train_one = np.append(train_acc_category, train_acc_total)
            result_train_one = np.append(result_train_one, train_acc_balanced).reshape(1,12)
            result_train = np.append(result_train, result_train_one, axis=0)
            result_valid_one = np.append(valid_acc_category, valid_acc_total)
            result_valid_one = np.append(result_valid_one, valid_acc_balanced).reshape(1,12)
            result_valid = np.append(result_valid, result_valid_one, axis=0)


        result_train_all_one = np.mean(result_train, axis=0)
        result_valid_all_one = np.mean(result_valid, axis=0)
        result_train_all = np.append(result_train_all, result_train_all_one.reshape(1,12), axis=0)
        result_valid_all = np.append(result_valid_all, result_valid_all_one.reshape(1,12), axis=0)

        add_df = pd.DataFrame([[result_train_all_one[0], result_valid_all_one[0]], \
                               [result_train_all_one[1], result_valid_all_one[1]], \
                               [result_train_all_one[2], result_valid_all_one[2]], \
                               [result_train_all_one[3], result_valid_all_one[3]], \
                               [result_train_all_one[4], result_valid_all_one[4]], \
                               [result_train_all_one[5], result_valid_all_one[5]], \
                               [result_train_all_one[6], result_valid_all_one[6]], \
                               [result_train_all_one[7], result_valid_all_one[7]], \
                               [result_train_all_one[8], result_valid_all_one[8]], \
                               [result_train_all_one[9], result_valid_all_one[9]], \
                               [result_train_all_one[10], result_valid_all_one[10]], \
                               [result_train_all_one[11], result_valid_all_one[11]]]).T

        add_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
        add_df.index   = ["Train:SEED" + str(seed), "Valid:SEED" + str(seed)]
        #pd.concat([df, add_df], axis=0)
        all_df = all_df.append(add_df)


    result_train_ave = np.mean(result_train_all, axis=0)
    result_train_max = np.max(result_train_all, axis=0)
    result_train_min = np.min(result_train_all, axis=0)
    result_valid_ave = np.mean(result_valid_all, axis=0)
    result_valid_max = np.max(result_valid_all, axis=0)
    result_valid_min = np.min(result_valid_all, axis=0)

    add_df = pd.DataFrame([[result_train_ave[0], result_train_max[0], result_train_min[0], result_valid_ave[0], result_valid_max[0], result_valid_min[0]], \
                           [result_train_ave[1], result_train_max[1], result_train_min[1], result_valid_ave[1], result_valid_max[1], result_valid_min[1]], \
                           [result_train_ave[2], result_train_max[2], result_train_min[2], result_valid_ave[2], result_valid_max[2], result_valid_min[2]], \
                           [result_train_ave[3], result_train_max[3], result_train_min[3], result_valid_ave[3], result_valid_max[3], result_valid_min[3]], \
                           [result_train_ave[4], result_train_max[4], result_train_min[4], result_valid_ave[4], result_valid_max[4], result_valid_min[4]], \
                           [result_train_ave[5], result_train_max[5], result_train_min[5], result_valid_ave[5], result_valid_max[5], result_valid_min[5]], \
                           [result_train_ave[6], result_train_max[6], result_train_min[6], result_valid_ave[6], result_valid_max[6], result_valid_min[6]], \
                           [result_train_ave[7], result_train_max[7], result_train_min[7], result_valid_ave[7], result_valid_max[7], result_valid_min[7]], \
                           [result_train_ave[8], result_train_max[8], result_train_min[8], result_valid_ave[8], result_valid_max[8], result_valid_min[8]], \
                           [result_train_ave[9], result_train_max[9], result_train_min[9], result_valid_ave[9], result_valid_max[9], result_valid_min[9]], \
                           [result_train_ave[10], result_train_max[10], result_train_min[10], result_valid_ave[10], result_valid_max[10], result_valid_min[10]], \
                           [result_train_ave[11], result_train_max[11], result_train_min[11], result_valid_ave[11], result_valid_max[11], result_valid_min[11]]]).T
    add_df.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "Total", "Balance"]
    add_df.index   = ["Train:Ave", "Train:Max", "Train:Min", "Valid:Ave", "Valid:Max", "Valid:Min"]
    df = df.append(add_df)
    df = df.append(train_df)
    df = df.append(valid_df)
    df = df.append(all_df)

    # 結果の配列をデータフレームに変換、CSVファイル出力
    #df = pd.DataFrame(result)
    filename = np.random.randint(1000000)
    print(filename)
    df.to_csv("result/result" + str(filename) + ".csv")



if __name__ == "__main__":
    train()
