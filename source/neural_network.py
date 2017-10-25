#coding:utf-8
import theano_function
import layers as convnet
import numpy as np
import pickle
import math
import theano

with open("parameter.pkl", 'rb') as file:
    params_dict = pickle.load(file)

########## Convolutional Neural Net ##########
class CNN( object ):

    def __init__( self, Layers ):

        func = theano_function.TheanoFunction( Layers )

        # layers - list of Layer instances
        self.Layers = Layers

        # theano functions
        self.output = func._func_forward()
        self.cost   = func._func_cost()
        self.train  = func._func_train()

        self.predict = func._func_predict()


# Conv-Pool-ReLu-Softmax
#def CPRS( train_data, network_list ):
def CPRS( network_list ):
    #print(train_data[0].shape)

    #Xdim = ( Xnch, Xrow, Xcol )
    ds1 = ( 3, 3 )
    st1 = ( 1, 1 )

    input = {}
    weight_dim = {}
    output = {}
    output_num = {}

    #入力パラメータ
    #input_dim = train_data[0].shape
    input_dim = ( params_dict["InputChannel"], params_dict["InputWidth"], params_dict["InputHeight"] )

    #weight_dim[0] = ( 16, input_dim[0], 5, 5 )
    #output_num[0] = 100
    #output_num[1] = 10
    reshape_flag = True
    #affine_cnt = 0

    layers = []
    gamma = {}
    beta = {}
    #for i in range( len( network_list ) ):
    for i in range( params_dict['LayerNum'] ):
        if network_list[i] == "BatchNorm":
            input[i] = output[i-1]
            output[i] = input[i]

            gamma[i] = theano.shared( np.ones( input[i], dtype = theano.config.floatX ) )
            beta[i] = theano.shared( np.zeros( input[i], dtype = theano.config.floatX ) )

            layers.append( convnet.BatchNormLayer( input[i], gamma[i], beta[i] ) )

        if network_list[i] == "Convolution":
            input[i] = input_dim if i == 0 else output[i - 1]
            weight_dim[i] = ( params_dict['Channel'+str(i+1)], input[i][0], params_dict['Width'+str(i+1)], params_dict['Height'+str(i+1)] )
            output[i] = ( weight_dim[i][0], input[i][1] - weight_dim[i][2] + 1, input[i][2] - weight_dim[i][3] + 1 )
            layers.append( convnet.ConvLayer( input[i], weight_dim[i] ) )

        #elif network_list[i] == "Pool":
        elif network_list[i] == "Pooling":
            input[i] = output[i - 1]
            output[i] = input[i]

            #if input[i][1] % 2 == 0:
                #output[i] = ( input[i][0], input[i][1] / 2, input[i][2] / 2 )
            #else:
                #output[i] = ( input[i][0], (input[i][1] / 2)+1, (input[i][2] / 2)+1 )

            layers.append( convnet.PoolLayer( input[i], output[i], ds1, st=st1, bias=True ) )

        elif network_list[i] == "Relu":
            input[i] = output[i - 1]
            output[i] = input[i]
            layers.append( convnet.ReluLayer() )

        elif network_list[i] == "Affine":
            if i == 0:
                input[i] = ( params_dict["InputChannel"], params_dict["InputWidth"], params_dict["InputHeight"] )
            else:
                input[i] = output[i - 1]
            if reshape_flag == True : input[i] = int( np.prod( input[i] ) )

            #output[i] = output_num[affine_cnt]
            output[i] = params_dict['Output'+str(i+1)]

            layers.append( convnet.AffineLayer( input[i], output[i], T4toMat = reshape_flag ) )
            reshape_flag = False
            #affine_cnt += 1

        elif network_list[i] == "Softmax":
            layers.append( convnet.SoftmaxLayer() )

    cnn = CNN( layers )
    return cnn
