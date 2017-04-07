# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:37:35 2013

@author: Ryan
"""

'''
 Convolutional Deep Belief Network (DBN)
'''

import sys, os
import numpy
import Image
from RBM import RBM, Trainer
from Convolutional import Convolutional, ConvolutionalTrainer
from utils import *
import datastorage
import pickle
import numpy as np
from mnist import MNIST
import threading
import itertools
import weka
from scipy.misc import imsave
from MyError import NaNError, MyError
import logging
import cProfile
import numpy.random as rng

running = True
a = None
def running_func():
    global running
    running = True
    while True:
        a = raw_input()
        if a == '':
            running = False
class ConvolutionalDBN(object):
    ''' Very simple class to conjoin Convolutional RBMs into layers. The init
        is not very useful. Mainly, relies on using the "addLayer" function
    '''
    def __init__(self, input=None, label=None,\
                 n_layers=0, num_filters=1, \
                 filter_shape=[30, 30], pool_shape=[2, 2]):
        
        self.x = input
        self.y = label

        self.rbm_layers = []
        self.rbm_trainers = []
        self.n_layers = n_layers
       
        assert self.n_layers >= 0
        
        # construct multi-layer
        for i in xrange(self.n_layers):

            # construct rbm_layer
            rbm_layer = Convolutional(num_filters, filter_shape, pool_shape)
            trainer = ConvolutionalTrainer(rbm_layer,.4,.01, .005)

            # add layer to lists
            self.rbm_layers.append(rbm_layer)
            self.rbm_trainers.append(trainer)
        self.n_layers = len(self.rbm_layers)

        self.recon_counter = 0
        

    def addLayer(self, rbm_layer, trainer=None):
        ''' Add a ConvRBM to network'''
        # add layer to lists
        self.rbm_layers.append(rbm_layer)
        self.rbm_trainers.append(trainer)
        self.n_layers = len(self.rbm_layers)

    def train(self, lr=0.1, cdk=1, epochs=100):
        ''' Train each layer, similar to the Convolutional.py test function'''
        global running, a
        print 'Starting thread'
        thr = threading.Thread(target=running_func)
        thr.start()
        # pre-train layer-wise
        print 'nlayers: ', self.n_layers
        for i in xrange(self.n_layers):

            learning_rate = lr
            print 'Getting training data...'
            if i == 0:
                training_data = self.x
                np.random.shuffle(training_data)
                training_data = training_data[0:20000]
            else:
                training_data = np.array([self.rbm_layers[i-1].pooled_sample(training_data[j]) for j in range(training_data.shape[0])])
            trainer = self.rbm_trainers[i]
            rbm = self.rbm_layers[i]
            minRE = -1
            avgRE = 9999
            print 'Training...'
            for epoch in xrange(epochs):
                np.random.shuffle(training_data)
                thr_list = []
                # For first layer, input are images [x,y]
                if i == 0:
                    iter_over = training_data.shape[0:-2]
                else:
                    # For upper layers, input are [k, x, y]
                    iter_over = training_data.shape[0:-3]
                ranges = map(range, iter_over)
                #for j in range(training_data.shape[0]):
                for j in itertools.product(*ranges):
                    #print 'j: ', j
                    mean_weights = abs(float(rbm.weights.mean()))
                    oldRE = avgRE
                    if i == 0: # i is the layer
                        #print training_data.shape
                        #print training_data[j].shape
                        
                        learning_rate = .99*learning_rate + .01*(mean_weights+epoch)/(10000 + epoch*epoch)
                        #trainer.learn(training_data[j], learning_rate, cdk=1+epoch/10)    
                        t = threading.Thread(target=trainer.learn, args = (training_data[j], learning_rate, 1+epoch/10,))
                        t.daemon = True
                        thr_list.append(t)
                        t.start()
                        avgRE = rbm.get_avg_error(training_data[j])
                        pass
                    else:
                        learning_rate = .99*learning_rate + .01*(mean_weights+epoch)/(10000 + epoch*epoch)
                        for k in range(training_data.shape[-3]):
                            t = threading.Thread(target=trainer.learn, args = (np.squeeze(training_data[j, k]), learning_rate, 1+epoch/10+i))
                            t.daemon = True
                            thr_list.append(t)
                            t.start()
                            
                            #print training_data[j, k].shape
                            #trainer.learn(np.squeeze(training_data[j, k]), learning_rate, cdk=1+epoch/10+i)
                            avgRE = rbm.get_avg_error(np.squeeze(training_data[j, k]))
                        
                    ''' Save minimum weights'''
                    if avgRE < minRE or minRE == -1:
                        minRE = avgRE
                        if minRE < oldRE and sum(j)%10==0:
                            print 'Saving...'
                            pickle.dump(rbm, open("minConvLayer" + str(i) + ".p", "wb"))
                            for k in range(rbm.num_filters):
                                imsave(os.path.join("Images", "weightsL"+str(i)+"_min_"+str(k)+".jpg"), rbm.weights[k])
                            print 'Done.'
                            self.reconstruct(self.x[0], i+1)
                    if avgRE <= oldRE:
                        direction = '-'
                        if abs(oldRE - avgRE) < (1 / (1+i)**3 ):
                            trainer.momentum = .8
                    else:
                        #learning_rate = .9*learning_rate + .1*(mean_weights)/5000000
                        direction = '+'
                        
                    if sum(j) % 3 == 0:
                            print i, 'Error ', str(epoch), ': ', avgRE, direction
                    if sum(j) % 400 == 0:                            
                            print 'Amt training data: ', len(training_data)
                            print i, 'Avg weights: ', str(rbm.weights.mean()),', Learning rate: ', str(learning_rate), ', Momentum: ', str(trainer.momentum)
                            print '    hid bias: ', str(rbm.hid_bias.mean()), ', vis bias: ', str(rbm.vis_bias)
                            #rbm.updateHistory()
                            #if sum(j) % 800:
                                #rbm.writeHistory(i)
                    for thr in thr_list:
                        thr.join()
                    thr_list = []
                    
                    if not running:
                        print 'First break'
                        print 'Breaking on running (out)'
                        print 'Epoch: ', str(epoch), ' iters: ', str(epoch*training_data.shape[0]+sum(j))
                        break
                if not running:
                    print 'Second break'
                    print 'Breaking on running (out)'
                    break
            print 'Saving layer i weights'
            pickle.dump(rbm, open("convLayer"+str(i)+".p", "wb"))      
            if minRE < avgRE:
                rbm = pickle.load(open("minConvLayer"+str(i)+".p", "rb"))
                self.rbm_layers[i] = rbm

            # Print weights to images
            for j in range(rbm.num_filters):
                imsave(os.path.join("Images", "weightsL"+str(i)+"_"+str(j)+".jpg"), rbm.weights[j])
            self.reconstruct(self.x[0], i+1)
            print 'Minimum reconstruction errors: ', minRE
            print 'nlayers: ', self.n_layers
            #if not running:
                #print 'Last break break'
                #print 'Breaking on running (out)'
                #break
            if 'a' in globals():
                del a
            running = True
        print 'Joining thread'
        thr.join()
    
    def pass_up(self, xin = None, num_Layers = None):
        ''' Feed image forward through network
        Needs to be fixed for pooled to hidden!!!!!!!!!!!!!!1111
        '''
        if xin is None:
            xin = self.x[0]
        if num_Layers is None:
            num_Layers = self.n_layers
        layer_output = xin
        for i in xrange(num_Layers):

            print layer_output.shape, '   up ', str(i)
            print 'layer: ', i
            if i == 0:
                layer_output = self.rbm_layers[i].pooled_sample(layer_output) #changed hidden to pooled, also below
            else:
                iter_over = layer_output.shape[0:-2]
                ranges = map(range, iter_over)
                #layer_output = np.array([self.rbm_layers[i].hidden_sample(layer_output[k]) for k in range(self.rbm_layers[i].num_filters)])
                #if i == num_Layers-1: # don't pool on last layer
                #    layer_output = np.array([self.rbm_layers[i].hidden_sample(layer_output[k]) for k in itertools.product(*ranges) ])
                #else:
                layer_output = np.array([self.rbm_layers[i].pooled_sample(layer_output[k]) for k in itertools.product(*ranges) ])                    
            #imsave('recon_up_'+str(i)+'.jpg', )
        assert len(layer_output.shape) == (num_Layers+2)
        return layer_output
        
    def pass_down(self, xin = None, num_Layers = None):
        ''' Feed image backwards through network
        Needs to be fixed for pooled to hidden!!!!!!!!!!!!!!!!!!!!!
        '''
        if xin is None:
            xin = self.x[0]
        if num_Layers is None:
            num_Layers = self.n_layers
        layer_output = xin
        for i in xrange(num_Layers-1, -1, -1): # this is right, counts down from nL-1 to 0
            print layer_output.shape, '   down ', str(i)
            
            #if i != num_Layers-1: # don't pool on last layer
            #added this line for pooling to hidden
            layer_output = self.rbm_layers[i].pooled_to_hidden(layer_output)
                
            if i == 0:
                print 'layer: ', i
                layer_output = self.rbm_layers[i].visible_expectation(layer_output)
            else:
                iter_over = layer_output.shape[0:-3]
                ranges = map(range, iter_over)
                print 'layer: ', i
                layer_output = np.array([self.rbm_layers[i].visible_sample(layer_output[k]) for k in itertools.product(*ranges) ])
        assert len(layer_output.shape) == 2
        return layer_output
    def predict():
        pass
    def reconstruct(self, xin = None, num_Layers = None):
        ''' Feed image through network, and back to visible layer
        '''
        hidden = self.pass_up(xin, num_Layers)
        reconstruction = self.pass_down(hidden, num_Layers)
        print reconstruction.shape, '  recon'
        imsave(os.path.join('Images', 'Reconstruction_Image'+str(self.recon_counter)+'.jpg'), reconstruction)
        self.recon_counter+=1
        return reconstruction

def loadNetwork():
    r = pickle.load(open("convLayer0.p", "rb"))
    r2 = pickle.load(open("convLayer1.p", "rb"))
    
    mn = MNIST()    
    training_data, classes = mn.load_training()
    training_data = np.asarray(training_data[0:10000])#[0:50])
    training_data = np.reshape(training_data, [len(training_data),28,28])
    
    cdbn = ConvolutionalDBN(training_data)
    cdbn.addLayer(r,t)
    cdbn.addLayer(r2,t2)
    
    return cdbn
    
def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
             finetune_lr=0.1, finetune_epochs=200):
    
    
    print 'Test ConvDBN'
    rLayers = 40
    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
    #r = Convolutional(rLayers, [12,12], [2,2], False) #Convolutional(2, [3, 3], [2, 2])
    r2 = Convolutional(rLayers, [6,6], [2,2], True)

    r = pickle.load(open("minConvLayer0.p", "rb"))
    #r2 = pickle.load(open("convLayer1.p", "rb"))
    rLayers = r.num_filters
    print rLayers
    '''Trainer(rbm, momentum=0., l2=0., target_sparsity=None):'''
    t = ConvolutionalTrainer(r,.5,0, .005)
    t2 = ConvolutionalTrainer(r2,.5,0, .003)
    
    ''' Get training data'''
    #training_data, classes =  datastorage.getData(["C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\female\\Zoom\\F"])

    mn = MNIST()    
    training_data, classes = mn.load_training()
    training_data = np.asarray(training_data[0:20000])#[0:50])
    training_data = np.reshape(training_data, [len(training_data),28,28])
    #print training_data.shape
    #training_data = np.array([np.asarray(Image.fromarray(training_data[i]).resize((10,10), Image.ANTIALIAS)) for i in range(len(training_data))])
    #print training_data.shape
    # construct DBN def __init__(self, input=None, label=None, n_layers=0, num_filters=1, 
        #                              filter_shape=[30, 30], pool_shape=[2, 2]):
    cdbn = ConvolutionalDBN(training_data)
    cdbn.addLayer(r,t)
    cdbn.addLayer(r2,t2)

    # pre-training (TrainUnsupervisedDBN) LR, CDK, EPOCh
    cdbn.train(0.000001, cdk=3, epochs=100)
    
    # test
    print 'Test...'
    training_data, classes = mn.load_training()
    training_data = np.asarray(training_data[0:10000])#[0:50])
    training_data = np.reshape(training_data, [len(training_data),28,28])
    cdbn.reconstruct(training_data[0])
    
    ###################################################################
    #print dbn.predict(x)
    print 'Weka....'
# Getting TRAINING DATA TO FILE
    amt_training = 1000
    tort = "TRAINING"
    mn = MNIST()    
    data, classes = mn.load_training()
    data = np.asarray(data)#[0:50])
    data = np.reshape(data, [len(data),28,28])
    # Change to numpy array
    classes = np.fromiter(classes, dtype=np.int)
    
    assert len(data) == len(classes)    
    p = np.random.permutation(len(data))
    data, classes = data[p], classes[p]
    
    testing_data = data[0:100]
    testing_classes = classes[0:100]
    
    training_data = data[100:100+amt_training]
    training_classes = classes[100:100+amt_training]

    weka_file = tort+"_10F_output.arff"
    wk = weka.Weka(relation="Number")
    last_percent, flag = -1, True
#
    print 'Writing Weka file: ', weka_file
    for i in range(len(training_classes)):
        out1 = r.pooled_expectation(training_data[i] )
        #print 'out1 size: ', out1.shape
        features = np.array([r2.pooled_expectation(out1[k]) for k in range(rLayers)])
        
        features = np.append(out1, features)
        features = features.ravel()
        classnum = training_classes[i]
        if flag:            
            f_shape = features.shape[0]
            flag = False
            for j in range(f_shape):
                wk.add_attribute('dl'+ str(j+1), 'NUMERIC')
            print 'Feature length: ', features.shape[0]
        bob = ''
        
        for j in features:
            bob += str(j)+','
            
        wk.add_data(bob + str(classnum))
        percent = int(100*float((i+1))/len(training_classes))
        if percent > last_percent:
            print 'Percent through samples: ', str(percent), '%'
        last_percent = percent
    wk.add_attribute('class', "{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}")
    wk.write_file(weka_file)
    print 'Finished!'

    tort = "TESTING"

    weka_file = tort+"_10F_output.arff"
    wk = weka.Weka(relation="Gender")
    last_percent, flag = -1, True

    print 'Writing Weka file: ', weka_file
    for i in range(len(testing_classes)):
        out1 = r.pooled_expectation(testing_data[i] )
        #print 'out1 size: ', out1.shape
        features = np.array([r2.pooled_expectation(out1[k]) for k in range(rLayers)])
        features = np.append(features, out1)
        features = features.ravel()
        classnum = testing_classes[i]
        if flag:            
            f_shape = features.shape[0]
            flag = False
            for j in range(f_shape):
                wk.add_attribute('dl'+ str(j+1), 'NUMERIC')
            print 'Feature length: ', features.shape[0]
        bob = ''
        
        for j in features:
            bob += str(j)+','
            
        wk.add_data(bob + str(classnum))
        percent = int(100*float((i+1))/len(testing_classes))
        if percent > last_percent:
            print 'Percent through samples: ', str(percent), '%'
        last_percent = percent
    wk.add_attribute('class', "{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}")
    wk.write_file(weka_file)
    print 'Finished!'
if __name__ == "__main__":
    cProfile.run('test_dbn()')
    #test_dbn()
