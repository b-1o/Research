#coding:utf-8
import numpy as np
import scipy as sp
import os
import cPickle
#import _pickle as cPickle


class CIFAR10( object ):

    def __init__( self, dirname ):

        self.path = dirname
        #f_meta = open( os.path.join( self.path, 'batches.meta'), 'r' )
        f_meta = open( os.path.join( 'cifar-10-batches-py', 'batches.meta'), 'r' )
        self.meta = cPickle.load( f_meta )
        #self.meta = cPickle.load(f_meta, encoding='latin1')
        f_meta.close()
        self.nclass = len( self.meta['label_names'] )

        #print '##### CIFAR-10 #####'
        #print '# label_names =', self.meta['label_names']
        #print '# num_vis = ', self.meta['num_vis']


    def _loadBatch( self, fn ):

        p = os.path.join( self.path, fn )
        f = open( p, 'r' )
        d = cPickle.load( f )
        f.close()
        data   = d['data']   # 10000 x 3072 ( 3072 = 3 x 32 x 32 ), unit8s
        labels = d['labels'] # 10000-dim, in { 0, 1, ..., 9 }

        return data, np.array( labels )


    def _loadL( self ):

        fnList = [ 'data_batch_%d' % i for i in range( 1, 6 ) ]
        dataList, labelsList = [], []
        for fn in fnList:
            d, l = self._loadBatch( fn )
            dataList.append( d )
            labelsList.append( l )

        return np.vstack( dataList ), np.hstack( labelsList )


    def _loadT( self ):

        return self._loadBatch( 'test_batch' )


    ##### loading the data
    #
    def loadData( self, LT ):

        if LT == 'L':
            dat, lab = self._loadL()
        else:
            dat, lab = self._loadT()

        X = np.asarray( dat, dtype = float ).reshape( ( -1, 3, 32, 32 ) )
        t = np.zeros( ( lab.shape[0], self.nclass ), dtype = bool )
        for ik in range( self.nclass ):
            t[lab == ik, ik] = True

        return X, lab, t


    ##### generating the index of training & validation data
    #
    def genIndexLV( self, lab, seed = 0 ):

        np.random.seed( seed )
        idx = np.random.permutation( lab.shape[0] )
        idxV = np.zeros( lab.shape[0], dtype = bool )

        # selecting 1000 images per class for validation
        for ik in range( self.nclass ):
            i = np.where( lab[idx] == ik )[0][:1000]
            idxV[i] = True

        #idxL = -idxV
        idxL = ~idxV

        return idxL, idxV



if __name__ == "__main__":

    import cv2

    #dirCIFAR10 = '../140823-pylearn2/data/cifar10/cifar-10-batches-py'
    dirCIFAR10 = 'cifar-10-batches-py'
    cifar10 = CIFAR10( dirCIFAR10 )
    dataL, labelsL = cifar10._loadL()

    w = h = 32
    nclass = 10
    nimg = 10
    gap = 4

    width  = nimg * ( w + gap ) + gap
    height = nclass * ( h + gap ) + gap
    img = np.zeros( ( height, width, 3 ), dtype = int ) + 128

    for iy in range( nclass ):
        lty = iy * ( h + gap ) + gap
        idx = np.where( labelsL == iy )[0]
        for ix in range( nimg ):
            ltx = ix * ( w + gap ) + gap
            tmp = dataL[idx[ix], :].reshape( ( 3, h, w ) )
            # BGR <= RGB
            img[lty:lty+h, ltx:ltx+w, 0] = tmp[2, :, :]
            img[lty:lty+h, ltx:ltx+w, 1] = tmp[1, :, :]
            img[lty:lty+h, ltx:ltx+w, 2] = tmp[0, :, :]

    cv2.imwrite( 'hoge.png', img )
