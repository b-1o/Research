#coding:utf-8
import theano
import theano.tensor as T

class TheanoFunction( object ):

    def __init__( self, Layers ):
        self.Layers = Layers


    ### theano function for output computation
    def _func_forward( self ):
        train_flag = True
        X = T.tensor4()  # Ndat x Xnch x Xrow x Xcol
        Z = _forward( self.Layers, X, train_flag )

        return theano.function( [ X ], Z )

    def _func_predict( self ):
        train_flag = False
        X = T.tensor4()
        Z = _forward( self.Layers, X, train_flag )

        return theano.function( [ X ], Z )

    ### theano function for cost computation
    def _func_cost( self ):
        Z = T.matrix()  # N x K
        lab = T.ivector()  # N-dim
        num = T.ivector()

        #cost = _crossentropy( Z, lab )

        # Mnist
        #cost = _crossentropy( Z, lab ) * T.pow( T.log(4800 / num) + 1, 2 )
        # Cifar
        #cost = _crossentropy( Z, lab ) * T.pow( T.log(4048 / num) + 1, 2 )
        # Uniqlo
        cost = return_cost( Z, lab, num )

        #return theano.function( [ Z, lab ], cost )
        return theano.function( [ Z, lab, num ], cost )


    ### theano function for gradient descent learning
    def _func_train( self ):
        X    = T.tensor4( 'X' )
        lab  = T.ivector( 'lab' )
        eta  = T.scalar( 'eta' )
        mu   = T.scalar( 'mu' )
        lam  = T.scalar( 'lambda' )
        num  = T.ivector( 'num' )

        train_flag = True

        Z = _forward( self.Layers, X, train_flag )
        #cost = T.mean( _crossentropy( Z, lab ) * T.pow( 1, 2 ) )

        # Mnist
        #cost = T.mean( _crossentropy( Z, lab ) * T.pow( T.log(4800 / num) + 1, 2 ) )
        # Cifar
        #cost = T.mean( _crossentropy( Z, lab ) * T.pow( T.log(4048 / num) + 1, 2 ) )
        # Uniqlo

        cost = T.mean( return_cost( Z, lab, num ) )

        updatesList = []
        for il, layer in enumerate( self.Layers ):

            # PoolLayer doesn't have W & dW
            #if not isinstance( layer, PoolLayer ) and layer.W != -1:
            if layer.weight:
                gradW = T.grad( cost, layer.W )
                #dWnew = -eta * gradW + mu * layer.dW
                dWnew = -eta * ( gradW + lam * layer.W ) + mu * layer.dW
                Wnew  = layer.W + dWnew
                updatesList.append( ( layer.W, Wnew ) )
                updatesList.append( ( layer.dW, dWnew ) )
            if layer.bias:
                gradb = T.grad( cost, layer.b )
                # no weight decay for bias
                dbnew = -eta * gradb + mu * layer.db
                bnew  = layer.b + dbnew
                updatesList.append( ( layer.b, bnew ) )
                updatesList.append( ( layer.db, dbnew ) )


            if layer.gamma:
                gradBNgamma = T.grad(cost, layer.BNgamma )
                dBNgamma_new = -eta * gradBNgamma + mu * layer.dBNgamma
                BNgamma_new = layer.BNgamma + dBNgamma_new
                updatesList.append( ( layer.BNgamma, BNgamma_new ) )
                updatesList.append( ( layer.dBNgamma, dBNgamma_new ) )
            if layer.beta:
                gradBNbeta = T.grad( cost, layer.BNbeta )
                dBNbeta_new = -eta * gradBNbeta + mu * layer.dBNbeta
                BNbeta_new = layer.BNbeta + dBNbeta_new
                updatesList.append( ( layer.BNbeta, BNbeta_new ) )
                updatesList.append( ( layer.dBNbeta, dBNbeta_new ) )

                updatesList.append( ( layer.BNmu, layer.dBNmu ) )
                updatesList.append( ( layer.BNsig2, layer.dBNsig2 ) )


        return theano.function( [ X, lab, eta, mu, lam, num ], cost, updates = updatesList )
        #return theano.function( [ X, lab, eta, mu, lam ], cost, updates = updatesList )



# feed forward
def _forward( Layers, X, train_flag ):
    Zprev = X
    for layer in Layers:
        Z = layer.output(Zprev, train_flag )
        Zprev = Z
    return Z


# cost function
def _crossentropy( Z, lab ):
    return T.nnet.categorical_crossentropy( Z, lab )

def return_cost( Z, lab, num ):
    return _crossentropy( Z, lab ) * (T.log(num / num) + 1)
    #return _crossentropy( Z, lab )
