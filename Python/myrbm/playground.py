# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 11:20:12 2014

@author: Ryan
"""

from RBM import RBM, Trainer
from Convolutional import Convolutional, ConvolutionalTrainer
from utils import *
from PIL import Image
from scipy.misc import imsave
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import *
from MyError import NaNError, MyError
import numpy as np
import logging
import threading
import pickle
import sys, os
import weka
import random
import cProfile
import numpy.random as rng
from datastorage import Data #dl_data
from mnist import MNIST
running = True
image_counter =2
def running_func():
    global running
    while True:
        a = raw_input()
        if a == '':
            running = False
learning_rate = .0002
oldE, avgE = 90., 10.
oldRE, avgRE, = float("inf"), float("inf")
minRE1, minRE2 = float("inf"), float("inf")
lock = threading.Lock()


def testConvolution():
    ''' Second thread used to break out of training loop'''
    thr = threading.Thread(target=running_func)
    thr.start()
    global running
    
    
    ''' Get training data'''
    #training_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\female\\Zoom\\F"])
    mn = MNIST()    
    training_data, classes = mn.load_training()
    training_data = np.asarray(training_data)#[0:50])
    training_data = np.reshape(training_data, [len(training_data),28,28])
    #training_data = training_data/training_data.max()
    #imsave("Images\\reconL0_"+str(20)+".jpg", training_data[0])
    
    print 'Test ConvRBM'
    rLayers = 40
    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
    #r = Convolutional(rLayers, [12,12], [2,2], False) #Convolutional(2, [3, 3], [2, 2])
    #rprev = pickle.load(open("minConvLayer1.p", "rb"))
    #print rprev.visible_to_pooled(training_data[0]).shape
    
    #hidden_data = rprev.visible_to_pooled(training_data[0])
    #for j in range(training_data.shape[0]):
    #    hidden_data = np.append(hidden_data, rprev.visible_to_pooled(training_data[j])[0:1], axis=0 )
    #training_data = hidden_data
    #print training_data.shape
    # Layer 2
    #r = Convolutional(rLayers, [6, 6], [2, 2], True) #pickle.load(open("convLayer1.p", "rb"))#
    #r.setUpperLayer(0, r)
    r = pickle.load(open("minConvPBS.p", "rb"))
    t = ConvolutionalTrainer(r, .5, 0, .003)
    #t.setUpperLayer()
    
    
    
    '''Trainer(rbm, momentum=0., l2=0., target_sparsity=None):'''
    #t = ConvolutionalTrainer(r,.5, 0, .005) #changed from .005 to .05
    saveType = "Serverlayer1"
    rLayers = r.num_filters
    print 'Training...'
    for i in range(rLayers):
        imsave(os.path.join("Images", "weightsL" +saveType+"_"+str(i)+".jpg"), r.weights[i])
    ''' Training for first layer'''
    for i in range(50):
            ''' Get NEW training data'''
            #        def trainThread():
            global avgRE, learning_rate, minRE1
            np.random.shuffle(training_data)
            for j in range(training_data.shape[0]):
                oldRE = avgRE
                
                ''' Slowly decrease learning rate from ~1/500th of the mean weight'''
                learning_rate = .99*learning_rate + .01*(abs(float(r.weights.mean()))/(100000 + i*i) )
                t.learn(training_data[j], learning_rate, cdk=2)
                avgRE = r.get_avg_error(training_data[j])
                
                # If error stops decreasing over 100 iterations, break loop
                if j+i*(training_data.shape[0]) % 9999 == 1:
                    oldRE = avgRE
                    
                ''' Save minimum weights'''
                if avgRE < oldRE:
                    direction = '-'
                    
                    if avgRE < minRE1:
                        minRE1 = avgRE
                        
                        if j % 100 == 0 :
                            ''' Reconstruct image for one layer'''                    
                            minRecon = r.reconstruct(training_data[j], 2)
                            #minRecon = minRecon / minRecon.max() * 255
                            with lock:
                                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_0.jpg"), training_data[j])
                                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_1.jpg"), minRecon)
                        
                        if j % 5 ==4: #minRE1 < 2000 and
                            with lock:
                                print 'Saving...'
                                pickle.dump(r, open("minConvLayer"+saveType+".p", "wb"))
                                for k in range(rLayers):
                                    imsave(os.path.join("Images", "weightsL"+saveType+"_min_"+str(k)+".jpg"), r.weights[k])
                    if abs(oldRE - avgRE) < 10: 
                        t.momentum = .7
                else:
                    direction = '+'
                with lock:
                    print i, 'Error 1: ', avgRE, direction, ' old: ', oldRE
                #if abs(oldRE - avgRE) < .01:
                #    break
                if not running:
                    with lock:
                        print 'First break'
                        print 'Breaking on running (in)'
                    break
        #thrs = []
        #for tt in range(1):
        #print 'Starting threads...'
        #thr_train = threading.Thread(target=trainThread)
        #thr_train.daemon = True
        #thrs.append(thr_train)
        #thr_train.start()
        #print 'Started.'
        #[x.start() for x in thrs]
        #with lock:
        #print 'Joining threads...'
        #[x.join() for x in thrs]
        #thr_train.join()

        #if abs(oldRE - avgRE) < .0001:
        #    break
            if not running:
                print 'Second break'
                print 'Breaking on running (out)'
                break
                #print 'shape: ', r.hidden_sample(training_data[j]).shape
    #with lock:
    #    print 'Joining threads...'
    #thr_train.join()
    print 'Saving layer 1 weights'
    pickle.dump(r, open("convLayer.p", "wb"))
    
    ''' Use the min reconstruction error weights as layer 1'''
    #if minRE1 < avgRE and minRE1 != -1:
    #    r = pickle.load(open("minConvLayer1.p", "rb"))
        
    # Print weights to images
    for i in range(rLayers):
        imsave(os.path.join("Images", "weightsL20_"+str(i)+".jpg"), r.weights[i])
    #thr.join()
    #print 'joined.'
    
if __name__ == '__main__':
    #cProfile.run('testConvolution()')
    testConvolution()
    