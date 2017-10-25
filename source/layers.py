#coding:utf-8
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

########## Layer(super class) ##########
class Layer( object ):
    def __init__( self, weight, bias, gamma, beta, Wshape=None, Bshape=None, Wini=0.01, floatX = theano.config.floatX ):
        self.weight = weight
        self.bias   = bias
        self.gamma  = gamma
        self.beta   = beta

        if self.weight:
            self.W = theano.shared( np.array( Wini * np.random.standard_normal( Wshape ), dtype = floatX ) )
            self.dW = theano.shared( np.zeros( Wshape, dtype = floatX ) )
        if self.bias:
            self.b  = theano.shared( np.zeros( Bshape, dtype = floatX ) )
            self.db = theano.shared( np.zeros( Bshape, dtype = floatX ) )



########## Relu Layer ##########
class ReluLayer( Layer ):

    def __init__( self, weight=False, bias=False, gamma=False, beta=False ):
        Layer.__init__( self, weight, bias, gamma, beta )

    def output ( self, X, train_flag ):
        #return T.switch( X > 0, X, 0 )
        return T.nnet.relu(X)


########## Softmax Layer ##########
class SoftmaxLayer( Layer ):

    def __init__( self, bias=False, weight=False, gamma=False, beta=False ):
        Layer.__init__( self, weight, bias, gamma, beta )

    def output ( self, X, train_flag ):
        return T.nnet.softmax( X )


########## Convolution Layer ##########
class ConvLayer( Layer ):

    def __init__( self, Xdim, Wdim, bias=True, weight=True, gamma=False, beta=False, Heinit=True, floatX = theano.config.floatX ):
        Layer.__init__( self, weight, bias, gamma, beta, Wshape=Wdim, Bshape=Wdim[0] ,Wini=weight_init( Heinit, Xdim ) )
        self.Xshape = Xdim
        self.Wshape = Wdim


    def output( self, X, train_flag ):
        # X:  Ndat x Xshape,  Y:  Ndat x Yshape
        Xs = ( None, self.Xshape[0], self.Xshape[1], self.Xshape[2] )
        Ws = self.Wshape
        Z = T.nnet.conv.conv2d( X, self.W, image_shape = Xs, filter_shape = Ws )
        if self.bias : Z += self.b.dimshuffle( 'x', 0, 'x', 'x' ) # 1 x nch x 1 x 1

        return Z


########## Pooling Layer ##########
class PoolLayer( Layer ):

    def __init__( self, Xdim, Ydim, ds, bias=False, weight=False, gamma=False, beta=False, st = None, floatX = theano.config.floatX ):
        Layer.__init__( self, weight, bias, gamma, beta, Bshape=Xdim[0] )

        # parameters of the pooling layer
        self.ds = ds
        self.st = st
        self.ignore_border = True
        self.pad = (1, 1)



    def output( self, X, train_flag ):
        Z = pool_2d(input = X, ws = self.ds, ignore_border = self.ignore_border, stride = (1,1), pad = (1,1), mode = 'max')
        if self.bias : Z += self.b.dimshuffle( 'x', 0, 'x', 'x' ) # 1 x nch x 1 x 1

        return Z


########## Affine Layer ##########
class AffineLayer( Layer ):

    def __init__( self, Din, Nunit, bias=True, weight=True, Heinit=True, gamma=False, beta=False, floatX = theano.config.floatX, T4toMat = False ):
        Layer.__init__( self, weight, bias, gamma, beta, Wshape=( Nunit, Din ), Bshape=Nunit, Wini=weight_init( Heinit, Din ) )
        self.T4toMat = T4toMat

        self.dropout = 0.5
        self.Nunit = Nunit
        #self.mask = T.shared_randomstreams.RandomStreams(0).uniform(( Nunit, )) <= self.dropout


    def output( self, X, train_flag ):
        if self.T4toMat : X = X.reshape( ( X.shape[0], -1 ) )
        Z = T.dot( X, self.W.T )
        if self.bias : Z += self.b


        mask = T.shared_randomstreams.RandomStreams(0).uniform(( self.Nunit, )) <= self.dropout

        if train_flag:
            return Z * mask
        else:
            return Z * self.dropout


class BatchNormLayer( Layer ):

    def __init__( self, Xdim, momentum=0.9, running_mean=None, running_var=None, bias=False, weight=False, gamma=True, beta=True, floatX = theano.config.floatX ):
        #gamma=False
        #beta=False
        Layer.__init__( self, weight, bias, gamma, beta )
        # BNa : gamma, BNb : beta
        #self.BNgamma  = theano.shared( np.ones( Xdim, dtype = floatX ) )
        #self.dBNgamma = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        #self.BNbeta  = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        #self.dBNbeta = theano.shared( np.zeros( Xdim, dtype = floatX ) )

        self.BNmu = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        self.dBNmu = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        self.BNsig2 = theano.shared( np.ones( Xdim, dtype = floatX ) )
        self.dBNsig2 = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        self.BNeps = 0.01

        self.BNgamma = theano.shared( np.ones( Xdim, dtype = floatX ) )
        self.BNbeta  = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        self.dBNgamma = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        self.dBNbeta  = theano.shared( np.zeros( Xdim, dtype = floatX ) )

        self.BNmean  = theano.shared( np.zeros( Xdim, dtype = floatX ) )
        self.BNvar   = theano.shared( np.ones( Xdim, dtype = floatX ) )


    def output( self, X, train_flag ):
        X_tmp = X

        if train_flag:
            #Z, _, _, self.BNmean, self.BNvar = T.nnet.bn.batch_normalization_train( X, gamma=self.BNgamma, beta=self.BNbeta, running_mean=self.BNmean, running_var=self.BNvar )
            BNmu = T.mean( X, axis=0 )
            X -= BNmu
            BNsig2 = T.mean( T.sqr(X), axis=0 ) + self.BNeps
            X /= T.sqrt( BNsig2 )
            self.dBNmu = BNmu
            self.dBNsig2 = BNsig2
            self.BNmu   = 0.9 * self.BNmu   + (1-0.9) * BNmu
            self.BNsig2 = 0.9 * self.BNsig2 + (1-0.9) * BNsig2

            #Z, _, _ = T.nnet.bn.batch_normalization_train( inputs=X_tmp, gamma=self.BNgamma,  beta=self.BNbeta )
            #print(a)
            #Z = a
            Z = self.BNgamma * X + self.BNbeta


        else:
            Z = T.nnet.bn.batch_normalization_test( X, gamma=self.BNgamma, beta=self.BNbeta, mean=self.BNmean, var=self.BNvar )
            #Z = T.nnet.bn.batch_normalization_test( inputs=X, gamma=self.BNgamma, beta=self.BNbeta, mean=self.BNmean, var=self.BNvar )
            #Z = self.BNa * ( X - self.BNmu ) / T.sqrt( self.BNsig2 ) + self.BNb


        return Z




def weight_init( Heinit, Xdim ):
    return np.sqrt( 2.0 / np.prod( Xdim ) ) if Heinit == True else 0.01
