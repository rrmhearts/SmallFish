# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:34:19 2013

@author: Ryan
"""

import os
import struct
import numpy as np
#import matplotlib.pyplot as plt
from scipy.misc import imsave

from array import array

class MNIST(object):
    def __init__(self, path=os.path.join(os.getcwd(), 'MNIST')):
        self.path = path
        #print "HERE: ", os.getcwd()

        self.test_img_fname = 't10k-images.idx3-ubyte'
        self.test_lbl_fname = 't10k-labels.idx1-ubyte'

        self.train_img_fname = 'train-images.idx3-ubyte'
        self.train_lbl_fname = 'train-labels.idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                         os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        #print self.train_img_fname
        #print self.train_lbl_fname
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                         os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod #Ignores instance, applies to Class of instance (cls)
    def load(cls, path_img, path_lbl): 
      
        with open(path_lbl, 'rb') as file:
            #print 'Labels stuff...'
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            #print 'Imgs stuff..'
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        #print 'size: ', size, 'rc: ', rows, ', ', cols
        for i in xrange(size):
            images.append([0]*rows*cols)

        for i in xrange(size):
            #print 'Here...'
            images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

        return images, labels

    def test(self):
        test_img, test_label = self.load_testing()
        train_img, train_label = self.load_training()
        
        train_arr = np.asarray(train_img)
        train_arr = np.reshape(train_arr, [len(train_arr),28,28])
        #imsave('bob.jpg',train_arr[0])
        print train_arr.shape, ' ', train_arr.max(), ' ', train_arr.min(), ' ', train_arr[20]
        assert len(test_img) == len(test_label)
        assert len(test_img) == 10000
        assert len(train_img) == len(train_label)
        assert len(train_img) == 60000
        print 'Showing num:%d' % train_label[0]
        print self.display(train_img[0])
        print
        return True

    @classmethod
    def display(cls, img, width=28):
        
        render = ''
        for i in range(len(img)):
            #print 'In display...'
            if i % width == 0: render += '\n'
            if img[i] > 200:
                render += '1'
            else:
                render += '0'
        return render

if __name__ == "__main__":
    print 'Testing'
    mn = MNIST(path=os.getcwd()+'\\MNIST')#'.')
    if mn.test():
        print 'Passed'