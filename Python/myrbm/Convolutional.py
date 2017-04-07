# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:16:26 2013

@author: Ryan based on Leif's wrong implementation
"""
from RBM import RBM, Trainer
from utils import *
from PIL import Image
from scipy.misc import imsave
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import *
from MyError import NaNError, MyError
import numpy as np
import logging, time
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
learning_rate = 10 #.02
oldE, avgE = 90., 10.
oldRE, avgRE, = float("inf"), float("inf")
minRE1, minRE2 = float("inf"), float("inf")
lock = threading.Lock()

class Convolutional(RBM):
    '''
    '''

##     # Binary True works, binary false does not work
    def __init__(self, num_filters, filter_shape, pool_shape, binary=True, scale=0.001):
        ''' Initialize a convolutional restricted boltzmann machine.

            num_filters: The number of convolution filters.
            filter_shape: An ordered pair describing the shape of the filters.
            pool_shape: An ordered pair describing the shape of the pooling groups.
            binary: True if the visible units are binary, False if the visible units
                are normally distributed.
            scale: Scale initial values by this parameter.
        '''
        self.num_filters = num_filters

        self.entropy = -1
        self.error = -1
        # Three-dim [num_filters, row_shape, col_shape]
        
        ''' Initiallize random weights'''
        self.weights = scale * rng.randn(num_filters, *filter_shape)
        self.vis_bias = scale * rng.randn(1)                 # Scalar visual bias
        
        # Bias array for hidden units
        ''' Bias for each hidden filter'''
        self.hid_bias = 2 * scale * rng.randn(num_filters)   # Scalar bias for each filter

        # Suggest visualizations for learning
        self.weights_history = np.expand_dims(self.weights, axis=0)
        self.last_hidden = None
        # Identity should be gaussian
        self._visible = binary and sigmoid or identity # gaussian #identity
        #self._visible = binary and sigmoid or gauss
        self._pool_shape = pool_shape
        #print 'Binary: ', binary
        #print "num filters: ", self.num_filters
        #print "filter size: ", self.weights.shape[1:]
        #print self.hid_bias.shape

    def _pool(self, hidden, down_sample=False):
        ''' Denominator of Equations 14 and 15 (sum elements in Beta)
            Given activity in the hidden units, pool it into groups.
            
            hidden: data to be pooled into groups of size _pool_shape. It is 
                assumed that the data is exp(I(h)) in the paper
            down_sample: determines if resulting image is of original shape or
                proportionally downsampled for each _pool_shape region.
        '''
        self.last_hidden = hidden.shape[-2:]
        # Convolved weights over visible image for each filter, hidden [0,1)
        k, r, c = hidden.shape
        # Pooling filter shape
        rows, cols = self._pool_shape
        
        # Active is 3-dim [x_shape, y_shape, k filters]
        active = hidden.T # Was the above line
        # Initialize pool layer to zeros
        # active is c, r, k
        ''' Downsample pooling is to reduce the image size'''
        if down_sample:
            pool = np.zeros((np.ceil(float(c)/cols),np.ceil(float(r)/rows), k), float)
        else:
            ''' Image size remains the same'''
            pool = np.zeros(active.shape, float)
        # Number of times you can apply pooling filter across columns
        for j in range(int(np.ceil(float(c) / cols))):  
            # Star and end points for filter in col-space
            cslice = slice(j * cols, (j + 1) * cols)
            # Number of times you can apply pooling filter across rows
            for i in range(int(np.ceil(float(r) / rows))):
                # Conjoin col-space slice and row-space slice (x,y)
                mask = (cslice, slice(i * rows, (i + 1) * rows))
                if down_sample:
                    #print active[mask].sum(axis=0).sum(axis=0)
                    pool[j,i] = active[mask].sum(axis=0).sum(axis=0)
                else:
                    pool[mask] = active[mask].sum(axis=0).sum(axis=0)
        # Pool.T is [k,row,col]
        #print pool.T.shape, ' ', hidden.shape
        assert pool.T.shape <= hidden.shape 
        return pool.T
        
    def set_kmeans_weights(self, data):
        ''' Given data, break the data into patches of size weights[1:] and then
            use kmeans to cluster patches to num_filters cluster points. If two
            points converge to one point, reduce the number of filters being used
            and update object properties accordingly.
            
            data: some training data to be broken into cluster points
        
        '''
        l, r, c = data.shape
        data = data / data.max()
        print "Formatting..."
        # Weight size
        rows, cols = self.weights.shape[1:]
        samples = rng.randn(1, rows, cols)
        #print "rc", rows, "  ", cols
        for k in range(l):
            # Force size to be [1, rows, cols]
            active = data[k:(k+1)]
            # Number of times you can apply pooling filter across columns
            for j in range(c/cols):  
                # Star and end points for filter in col-space
                cslice = slice(j * cols, (j + 1) * cols)
                # Number of times you can apply pooling filter across rows
                for i in range(r/rows):
                    # Conjoin col-space slice and row-space slice (x,y)
                    mask = (slice(0,1), slice(i * rows, (i + 1) * rows), cslice)
                    ''' If remaining portion of image is large enough'''
                    if active[mask].shape[1] >= rows and active[mask].shape[2] >= cols:
                        ''' Append masked region to samples array'''
                        samples = np.concatenate((samples, active[mask]), axis=0)
                    #print samples.shape
        # Mildly blur image
        samples = gaussian_filter(samples, 1)
        # Flatten for kmeans
        samples = np.reshape(samples, [len(samples), rows*cols])
        print "Running kmeans..."
        centroids, variance = kmeans(samples, self.num_filters)
        centroids = np.reshape(centroids, [len(centroids), rows, cols])
        
        # Initialize object for new weights
        self.weights = centroids
        self.num_filters = self.weights.shape[0]
        self.hid_bias = self.hid_bias[0:self.num_filters]
        #print "num filters: ", self.num_filters
        #print "filter size: ", self.weights.shape[1:]
        #print self.hid_bias.shape
        
    def hidden_max_pooling(self, visible, bias=0.):     # Visible is [row, col]
        ''' Given visible data, return the max pooling unit values after softmax.
            This is similar to traditional Convolutional Networks max pooling and 
            is not currently being used
            
            visible: input data to go to max pooling
        '''
        
        
        active = np.array([
            # convolve 2-D visible with one 2-D filter
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid')
            # Notice there are TWO transposes
            for k in range(self.num_filters)]).T + self.hid_bias + bias
        active = softmax(active.T)
        l, r, c = active.shape
        # Pooling filter shape
        rows, cols = self._pool_shape
        active = active.T # Was the above line
        pool = np.zeros((np.ceil(float(c)/cols),np.ceil(float(r)/rows),l), float)
        # Number of times you can apply pooling filter across columns
        for j in range(int(np.ceil(float(c) / cols))):  
            # Star and end points for filter in col-space
            cslice = slice(j * cols, (j + 1) * cols)
            # Number of times you can apply pooling filter across rows
            for i in range(int(np.ceil(float(r) / rows))):
                # Conjoin col-space slice and row-space slice (x,y)
                mask = (cslice, slice(i * rows, (i + 1) * rows))
                pool[j,i] = active[mask].max(axis=0).max(axis=0)
        assert pool.T.shape <= active.T.shape 
        return pool.T
    
    def pooled_to_hidden(self, pool):
        ''' Propagate down from layer k pool to layer k hidden.
            NOT FINISHED        
        '''
        # Ensure 1s and 0s
        px, py = pool.shape[-2:]
        pk = pool.shape[0:-2]
        #print 'pshape: ', pool.shape
        # px == ceil(hidden/xmlt)
        xmlt, ymlt = self._pool_shape
        hidden = np.zeros( pk + self.last_hidden ).T

        print 'hidden shape: ', hidden.shape
        # 28x28 --> 39,(16,16) --> 39,9,9 ???
        pool = bernoulli(pool).T
        #print 'hshape: ', hidden.shape, ' pshape: ', pool.shape
        #print 'field shape: ', self._pool_shape
        assert ymlt*pool.shape[0] == hidden.shape[0] or ymlt*pool.shape[0] == hidden.shape[0]+1
        assert xmlt*pool.shape[1] == hidden.shape[1] or xmlt*pool.shape[1] == hidden.shape[1]+1
        
        for y in range(0, hidden.shape[0]-ymlt, ymlt):#pool.shape[0]):
            for x in range(0, hidden.shape[1]-xmlt, xmlt):#pool.shape[1]):
                # insert pool value into random location of hidden
                xrand = random.randint(0, xmlt-1)
                yrand = random.randint(0, ymlt-1)
                hidden[y+yrand,x+xrand,:] = pool[np.ceil(float(y+yrand)/ymlt),
                                    np.ceil(float(x+xrand)/xmlt)]
        return hidden.T
        
    def visible_to_hidden(self,visible, bias=0.):
        ''' Visible to hidden sample'''
        return bernoulli(self.hidden_expectation(visible))
        
    def visible_to_pooled(self, visible, bias=0):
        ''' Visible to pooled sample'''
        return bernoulli(self.pooled_expectation(visible))
    
    def pooled_sample(self, visible, bias=0.):
        '''Sample hidden layer'''
        return bernoulli( self.pooled_expectation(visible))
        
    def pooled_expectation(self, visible, bias=0.):     # Visible is [row, col]
        ''' Equation 15 in Lee paper
            Given visible data, return the expected pooling unit values.
        '''
        activation = np.array([
                # convolve 2-D visible with one 2-D filter
                convolve(visible, self.weights[k, ::-1, ::-1], 'valid')
                for k in range(self.num_filters)]).T + self.hid_bias + bias
        act = activation.T
        
        # Prevent overflow of data
        dtype = type(act.max())
        if act.max() < np.log(np.finfo(dtype).max #max datatype size
                / self._pool_shape[0]*self._pool_shape[1] -1):
            #  Unstable 1 - P(pool=0 | v)
            pool0 = 1. / (1. + self._pool(np.exp(act), True))
        else: #except OverflowError:
            act = act / act.max() * np.log(np.finfo(dtype).max 
                / self._pool_shape[0]*self._pool_shape[1] -1)
            pool0 = 1. / (1. + self._pool(np.exp(act), True))
        '''max_inve = np.exp(-act.max())'''
        '''pool0 = max_inve / (max_inve + self._pool(np.exp(act-act.max()), True))'''
        return 1. - pool0 #/pool0.max()

    def hidden_expectation(self, visible, bias=0.):
        '''Equation 14 in Lee paper
           Given visible data, return the expected hidden unit values.
        '''
        #For batch weights[k:k+1, ::-1, ::-1]
        activation = np.array([
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid') # flipped h & v
            for k in range(self.num_filters)]).T + self.hid_bias + bias
        act = activation.T
        
        act = prevent_overflow(act)

        act = np.exp(act)
        expectation = act / (1. + self._pool(act) )

        return expectation

    def visible_expectation(self, hidden, bias=0.):
        '''Equation 10 or 11 in Lee paper
           Given hidden states, return the expected visible unit values.
        '''
        activation = sum(
            convolve(hidden[k], self.weights[k], 'full')
            for k in range(self.num_filters)) + self.vis_bias + bias
        ''' _visible is defined in init() if binary RBM or continuous'''
        return self._visible(activation)
    def updateHistory(self):
        try:
            self.weights_history = np.concatenate((self.weights_history, 
                        np.expand_dims(self.weights, axis=0)), axis=0)
        except AttributeError:
            self.weights_history = np.expand_dims(self.weights, axis=0)
        
    def writeHistory(self, i):
        try:
            pickle.dump(self, open("backupNetwork.p", "wb"))
            pickle.dump(self.weights_history, open("weightsHistory_"+str(i)+".p", "wb"))
        except MemoryError:
            print "Memory error!"
        except IOError:
            print "He's dead, Jim!"
            pickle.dump(self, open("backupNetwork.p", "wb"))

            
        
class ConvolutionalTrainer(Trainer):
    ''' Training functions for Convolutional networks. Depends on functions
        implemented in Trainer in RBM.py
    '''
    global lock
    def learn(self, visible, learning_rate=0.2, cdk=1):
        ''' Calculate and apply gradients
            visible: data to learn
            learning_rate: How much gradient affects the weights
            cdk: How far to take contrastive divergece (1 is quick)
        '''
        #self.learning_rate = 0.99*self.learning_rate + 0.01*learning_rate
        gradients = self.calculate_gradients(visible,cdk)
        with lock:
            self.apply_gradients(*gradients, learning_rate=learning_rate)
        
    def calculate_gradients(self, visible, cdk=1):
        ''' Calculate gradients for an instance of visible data.
            Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

            visible: A single array of visible data.
        '''
        
        ''' Get hidden representations (h0,h1) and reconstructed visible (v1)'''
        passes = self.rbm.iter_passes(visible)
        v0, h0 = passes.next()
        # might need soft max on v0,h0, v1, h1
        for i in range(cdk):
            v1, h1 = passes.next()
            if i + 1 == cdk:
                break
        ''' Equation 17. Contrastive divergence for weights and biases'''
        gw = np.array([
            convolve(v0, h0[k, ::-1, ::-1], 'valid') -
            convolve(v1, h1[k, ::-1, ::-1], 'valid')
            for k in range(self.rbm.num_filters)])/h0.size
        
        ''' Equation 19. One visible bias'''
        #if not self.upper:        
        gv = (v0 - v1).sum()#/v0.size
        #else:
        #    gv = 0
            
        ''' Equation 18. One bias for each hidden filter'''
        gh = (h0 - h1).sum(axis=-1).sum(axis=-1)#/h0.size
 
        ''' Equation 16'''
        if self.target_sparsity is not None:
            self.sparseness = self.target_sparsity - self.rbm.hidden_expectation(visible).mean(axis=-1).mean(axis=-1)
        if self.sparseness is not None:
            gh += self.sparseness
        #print 'grads: ', gh.mean(), '    ',  gv.mean(), gw.mean()
        if not self.upper:
            logging.debug('displacement: %.3g, hidden activations: %.3g',
                      np.linalg.norm(gv), h0.mean(axis=-1).mean(axis=-1).std())
        else:
            logging.debug('displacement: %.3g, hidden activations: %.3g',
                      h0.mean(axis=-1).mean(axis=-1).std())
        #print 'gw: ', gw.shape
        #print gw.mean(axis=1).mean(axis=1).shape
        #gw = gw -  gw.mean(axis=0) #*gw.mean() gw.mean(axis=1).mean(axis=1) +
        return gw, gv, gh


def testConvolution():
    ''' Second thread used to break out of training loop'''
    
    # Time SO FAR: 9 days 22.34 hours
    # 0 days 0 hours running now..
    # 92 Epochs of all data
    # 0 running now..
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
    saveType = "1speed" #"2kmeans_4x4" #20
    
    np.random.seed(101)
    np.random.shuffle(training_data)
    
    training_data = (training_data[0:10000]*1.0)/255
    print 'Test ConvRBM'
    rLayers = 40
    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
    #r = Convolutional(rLayers, [12,12], [2,2], True) #Convolutional(2, [3, 3], [2, 2])
    #rprev = pickle.load(open("minConvPBS.p", "rb"))
    #print rprev.visible_to_pooled(training_data[0]).shape
    
    #hidden_data = np.expand_dims(rprev.visible_to_pooled(training_data[0])[0], axis=0)
    #print hidden_data.shape
    #print rprev.pooled_expectation(training_data[j]).shape
    #for j in range(training_data.shape[0]):
    #    for i in range(training_data.shape[1]):
    #       hidden_data = np.append(hidden_data, np.expand_dims(rprev.pooled_expectation(training_data[j])[i], axis=0), axis=0 ) #visible_to_pooled
           #            hidden_data = np.append(hidden_data, rprev.pooled_expectation(training_data[j])[0:1], axis=0 ) #visible_to_pooled

    #training_data = hidden_data
    
    print training_data.shape
    # Layer 2
    #r =Convolutional(rLayers, [6, 6], [2, 2], True)
    #r = pickle.load(open("minConvLayer"+saveType+".p", "rb"))#
    r = pickle.load(open("convLayer.p", "rb"))#
    #r =Convolutional(rLayers, [4, 4], [2, 2], True)
    #r.set_kmeans_weights(training_data[0:1000]/255/40)
    
    #r.setUpperLayer(0, r)
    #r = pickle.load(open("minConvLayer1.p", "rb"))
    t = ConvolutionalTrainer(r, .3, 0.00001, .01)
    #t.add_noise(.001)
    #r.error = 50
    #t.setUpperLayer()
    
    #r.set_kmeans_weights()
    #r._visible = identity
    #minRecon = 50
    r.error = 80
    learning_rate = 1
    '''Trainer(rbm, momentum=0., l2=0., target_sparsity=None):'''
    #t = ConvolutionalTrainer(r,.5, 0, .005) #changed from .005 to .05
    rLayers = r.num_filters
    print 'Training...'
    for i in range(rLayers):
        imsave(os.path.join("Images", "weightsL" +saveType+"_"+str(i)+".jpg"), r.weights[i])
        
    start = time.clock()
    ''' Training for first layer'''
    for i in range(1000):
            ''' Get NEW training data'''
            #        def trainThread():
            global avgRE, learning_rate, minRE1
            np.random.shuffle(training_data)
            for j in range(training_data.shape[0]):
                oldRE = avgRE
                
                ''' Slowly decrease learning rate from ~1/500th of the mean weight'''
                learning_rate = .99*learning_rate + .01*(abs(float(r.weights.mean()))/(1000) )
                t.learn(training_data[j], learning_rate, cdk=1)
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
                                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_0.jpg"), training_data[j]*255)
                                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_1.jpg"), minRecon*255)
                        
                        if j % 5 ==4: #minRE1 < 2000 and
                            with lock:
                                print 'Saving...'
                                pickle.dump(r, open("minConvLayer"+saveType+".p", "wb"))
                                for k in range(rLayers):
                                    imsave(os.path.join("Images", "weightsL"+saveType+"_min_"+str(k)+".jpg"), r.weights[k])
                    if abs(oldRE - avgRE) < .1: 
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
    elapsed = (time.clock() - start)
    print "Elapsed time: ", elapsed
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
    print 'Done.'
    
if __name__ == '__main__':
    #cProfile.run('testConvolution()')
    testConvolution()
    

    
#def getData(loc = ["C:\\Users\\Ryan\\Documents\\SWAG_SUBSET\\male\\Zoom", "C:\\Users\\Ryan\\Documents\\SWAG_SUBSET\\female\\Zoom"], title="SWAG_data"):
#    ''' Get data from location, returns normalized data to [0,1] and the respective classes
#    '''
#    d = Data(loc)
#    input_data = d.get_people_set()
#    #print input_data[0][0].shape
#    #print input_data[0][1]
#    #print input_data[0][1][1]
#    
#    ''' Remove images that are not of the right size'''
#    for subject in input_data:
#        pic = subject[0]#input_data[i][0]
#        if pic.shape != (150,90,3):
#            print 'Removed an element'
#            input_data.remove(subject)#input_data[i]) #this broke?alueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
#            print 'New size: ', len(input_data)
#
#    ''' Pull correct images into data array'''
#    data = np.zeros((len(input_data), input_data[0][0].shape[0],input_data[0][0].shape[1]))
#    for i in range(len(input_data)):
#        pic = input_data[i][0]
#        #print type(pic), ' ', pic.shape
#        if pic.shape == (150,90,3):
#            #print 'success!'
#            data[i,:,:] = Image.fromarray(pic, 'RGB').convert('L')
#        else:
#            raise MyError(pic.shape)
#            #data[i,:,:] = data[i-1,:,:]
#    ''' Get classes from data'''
#    classes = [subject[1] for subject in input_data]
#    #print classes
#    pickle.dump([data, classes], open(title+".p", "wb"))
#    return data, classes
#
#def twoLayerReconstruction(data, r, r2):
#    ''' Attemped reconstruction of image'''
#    global image_counter
#    out1 = r.hidden_sample(data)
#    #print out1.shape
#    out2 = np.array([r2.hidden_sample(out1[k]) for k in range(r.num_filters)])
#    #print out2.shape
#    out1 = np.array([r2.visible_sample(out2[k]) for k in range(r.num_filters)])
#    #print out1.shape
#    recon = r.visible_expectation(out1)
#    recon = 255*(recon-recon.min())/recon.max()
#    #image_counter
#    imsave('Images\\2Layer_function_im'+str(image_counter)+'.jpg', recon)
#    image_counter+=1

#def testConvolution():
#    ''' Second thread used to break out of training loop'''
#    thr = threading.Thread(target=running_func)
#    thr.start()
#    global running
#    
#    
#    ''' Get training data'''
#    #training_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\female\\Zoom\\F"])
#    mn = MNIST()    
#    training_data, classes = mn.load_training()
#    training_data = np.asarray(training_data)#[0:50])
#    training_data = np.reshape(training_data, [len(training_data),28,28])
#    #training_data = training_data/training_data.max()
#    #imsave("Images\\reconL0_"+str(20)+".jpg", training_data[0])
#    
#    print 'Test ConvRBM'
#    rLayers = 40
#    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
#    #r = Convolutional(rLayers, [12,12], [2,2], False) #Convolutional(2, [3, 3], [2, 2])
#    #r.set_kmeans_weights(training_data[0:100]/255)
#
#    r2 = Convolutional(rLayers, [6,6], [2,2], True)
#
#    r = pickle.load(open("convLayer1.p", "rb"))
#    #r2 = pickle.load(open("minConvLayer2.p", "rb"))
#    
#    '''Trainer(rbm, momentum=0., l2=0., target_sparsity=None):'''
#    t = ConvolutionalTrainer(r,.5, 0, .004) #changed from .005 to .05
#    t2 = ConvolutionalTrainer(r2,.5,0, .004)
#    
#    #''' If TOO converged'''    
#    #t2.add_noise(1)
#    #r2.entropy=-1
#    
#
#    
#    rLayers = r.num_filters
#    
#
#    learning_rate = .3
#    entropy = 12.
#    oldE, avgE = 90., 10.
#    oldRE, avgRE, = float("inf"), float("inf")
#    minRE1, minRE2 = float("inf"), float("inf")
#    
#    print 'Training...'
#    for i in range(rLayers):
#        imsave("Images\\weightsL1_"+str(i)+".jpg", r.weights[i])
#    ''' Training for first layer'''
#    for i in range(1000):
#        ''' Get NEW training data'''
#        if i == 15:
#            pass
#            #print 'Loading new data...'
#            #training_data, _ =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\female\\Zoom\\F"])
#            
#        np.random.shuffle(training_data)
#        for j in range(training_data.shape[0]):
#            oldRE = avgRE
#            
#            ''' Slowly decrease learning rate from ~1/500th of the mean weight'''
#            learning_rate = .999*learning_rate + .001*(float(r.weights.mean()+i)/(10 + i*i*i) )
#            t.learn(training_data[j], learning_rate, cdk=1+min((i+1)*(j/20),30) )
#            entropy = r.get_entropy(training_data[j])
#            avgE = r.get_avg_entropy()
#            avgRE = r.get_avg_error(training_data[j])
#            
#            # If error stops decreasing over 100 iterations, break loop
#            if j+i*(training_data.shape[0]) % 9999 == 1:
#                if (oldRE-avgRE) < .001:
#                    print "old: ", oldRE, ", new: ", avgRE
#                    break
#                oldRE = avgRE
#                
#            ''' Save minimum weights'''
#            if avgRE < oldRE:
#                direction = '-'
#                
#                if avgRE < minRE1:
#                    minRE1 = avgRE
#                    
#                    if j % 20 == 0 :
#                        ''' Reconstruct image for one layer'''                    
#                        minRecon = r.reconstruct(training_data[j])
#                        print "minrecon stats: ", minRecon.max(), minRecon.min()
#                        #minRecon = minRecon / minRecon.max() * 255
#                        imsave("Images\\reconL1_"+str(i*100+j)+"_0.jpg", training_data[j])
#                        imsave("Images\\reconL1_"+str(i*100+j)+"_1.jpg", minRecon)
#                    
#                    if j % 5 ==4: #minRE1 < 2000 and
#                        print 'Saving...'
#                        pickle.dump(r, open("minConvLayer1.p", "wb"))
#                        for k in range(rLayers):
#                            imsave("Images\\weightsL1_min_"+str(k)+".jpg", r.weights[k])
#                        print 'Done.'
#                if abs(oldRE - avgRE) < 10: 
#                    t.momentum = .8
#            else:
#                direction = '+'
#            if j % 2 == 0:
#                print i, 'Entropy 1: ', avgE, ', error: ', avgRE, direction, ' ', oldRE
#            #if abs(oldRE - avgRE) < .01:
#            #    break
#            if not running:
#                print 'Breaking on running (in)'
#                break
#        #if abs(oldRE - avgRE) < .0001:
#        #    break
#        if not running:
#            print 'Breaking on running (out)'
#            break
#                #print 'shape: ', r.hidden_sample(training_data[j]).shape
#    print 'Saving layer 1 weights'
#    pickle.dump(r, open("convLayer1.p", "wb"))
#    
#    ''' Use the min reconstruction error weights as layer 1'''
#    if minRE1 < avgRE and minRE1 != -1:
#        r = pickle.load(open("minConvLayer1.p", "rb"))
#        
#    # Print weights to images
#    for i in range(rLayers):
#        imsave("Images\\weightsL1_"+str(i)+".jpg", r.weights[i])
#  
#    minRE2 = -1
#    running = True
#    learning_rate = .001
#    
#    print 'Creating hidden layers...'
#    hidden_data = r.visible_to_pooled(training_data[0])
#    print hidden_data.shape
#    for j in range(training_data.shape[0]):
#        hidden_data = np.append(hidden_data, r.visible_to_pooled(training_data[j]), axis=0 )
#    print hidden_data.shape
#    print 'Performing kmeans...'
#    #r2.set_kmeans_weights(hidden_data[0:100])
#    r2Layers = r2.num_filters
#    #Inf on error?
#    
#    print 'Training layer 2...'
#    '''Training for second layer'''
#    for i in range(40):
#        twoLayerReconstruction(training_data[0],r,r2)
#        #if i == 15:
#        #    print 'Loading new data...'
#            #training_data, _ =  getData()
#        #print 'Shuffling...'
#        np.random.shuffle(training_data)
#        '''Iterate through training data'''
#        for j in range(training_data.shape[0]):
#            oldRE = avgRE
#            
#            ''' Use (modified) pooling function to make second layer data input'''
#            hidden_units = r.visible_to_pooled(training_data[j])
#            #print 'shape: ', hidden_units.shape
#            #print 'shape: ', hidden_units[0].shape
#            '''For each hidden layer (K layers), learn from each one'''
#            for k in range(hidden_units.shape[0]):
#                learning_rate = .9*learning_rate + .1*(float(r.weights.mean()+i)/(500 + i*i) )
#                t2.learn(hidden_units[k], learning_rate, cdk=1+i/4)
#                    
#                entropy = r2.get_entropy(hidden_units[k])
#                avgE = r2.get_avg_entropy()
#                avgRE = r2.get_avg_error(hidden_units[k])
#                
#            ''' Save minimum weights'''
#            if avgRE < oldRE:
#                direction = '-'
#                if avgRE < minRE2 or minRE2 == -1:
#                    minRE2 = avgRE
#                    if minRE2 < 1000 and j % 5 == 4:
#                        print 'Saving...'
#                        pickle.dump(r2, open("minConvLayer2.p", "wb")) 
#                        print 'Done.'
#                if abs(oldRE - avgRE) < .5:
#                    t.momentum = .7
#            else:
#                direction = '+'
#                
#            if j % 5 == 0:
#                    print i, 'Entropy 2: ', avgE, ', error: ', avgRE, direction
#            #if entropy < .0003:
#            #    break
#            #if abs(oldRE - avgRE) < .01:
#            #    break
#            if not running:
#                print 'Break on not running (in)' 
#                break
#        #if entropy < .0003:
#        #    break
#        #if abs(oldRE - avgRE) < .001:
#        #    break            
#        if not running:
#            print 'Break on not running (out)' 
#            break
#    print 'Saving layer 2 weights'
#    pickle.dump(r2, open("convLayer2.p", "wb"))      
#    if minRE2 < avgRE:
#        r2 = pickle.load(open("minConvLayer2.p", "rb"))
#    
#    # Print weights to images
#    for i in range(r2Layers):
#        imsave("Images\\weightsL2_"+str(i)+".jpg", r2.weights[i])
#    
#    print 'Minimum reconstruction errors: ', minRE1, ', ', minRE2
#    
#    ''' Finished with network. Below this point is only saving the output feature 
#        vector to a Weka file for training data as well as test data
#    '''
#    #testing_data, classes =  getData(200)
#    # Getting TRAINING DATA TO FILE
#    tort = "TRAINING"
#    testing_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\female\\Zoom\\F"])
#
#    #testing_data = training_data
#
#    weka_file = tort+"_10F_output.arff"
#    wk = weka.Weka(relation="Gender")
#    last_percent, flag = -1, True
#
#    print 'Writing Weka file: ', weka_file
#    for i in range(len(classes)):
#        out1 = r.pooled_expectation(testing_data[i] )
#        #print 'out1 size: ', out1.shape
#        features = np.array([r2.pooled_expectation(out1[k]) for k in range(rLayers)])
#        features = features.ravel()
#        gender = classes[i]
#        if flag:            
#            f_shape = features.shape[0]
#            flag = False
#            for j in range(f_shape):
#                wk.add_attribute('dl'+ str(j+1), 'NUMERIC')
#            print 'Feature length: ', features.shape[0]
#        bob = ''
#        
#        for j in features:
#            bob += str(j)+','
#            
#        wk.add_data(bob + gender.lower())
#        percent = int(100*float((i+1))/len(classes))
#        if percent > last_percent:
#            print 'Percent through samples: ', str(percent), '%'
#        last_percent = percent
#    wk.add_attribute('class', "{male, female}")
#    wk.write_file(weka_file)
#    print 'Finished!'
#
#    tort = "TESTING"
#    testing_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\male\\Zoom", "C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\female\\Zoom"])
#
#    #testing_data = training_data
#
##    weka_file = tort+"_10F_output.arff"
##    wk = weka.Weka(relation="Gender")
##    last_percent, flag = -1, True
##
##    print 'Writing Weka file: ', weka_file
##    for i in range(len(classes)):
##        out1 = r.pooled_expectation(testing_data[i] )
##        #print 'out1 size: ', out1.shape
##        features = np.array([r2.pooled_expectation(out1[k]) for k in range(rLayers)])
##        features = features.ravel()
##        gender = classes[i]
##        if flag:            
##            f_shape = features.shape[0]
##            flag = False
##            for j in range(f_shape):
##                wk.add_attribute('dl'+ str(j+1), 'NUMERIC')
##            print 'Feature length: ', features.shape[0]
##        bob = ''
##        
##        for j in features:
##            bob += str(j)+','
##            
##        wk.add_data(bob + gender.lower())
##        percent = int(100*float((i+1))/len(classes))
##        if percent > last_percent:
##            print 'Percent through samples: ', str(percent), '%'
##        last_percent = percent
##    wk.add_attribute('class', "{male, female}")
##    wk.write_file(weka_file)
##    print 'Finished!'
#    
#    thr.join()
#    



#def kmeans2D(data, k, size):
#    l, r, c = data.shape
#    print "Processing data..."
#    # Weight size
#    rows, cols = size
#    
#    samples = rng.randn(1, rows, cols)
#    #print "rc", rows, "  ", cols
#    for k in range(l):
#        active = data[k:(k+1)]
#        # Number of times you can apply pooling filter across columns
#        for j in range(c/cols):  
#            # Star and end points for filter in col-space
#            cslice = slice(j * cols, (j + 1) * cols)
#            # Number of times you can apply pooling filter across rows
#            for i in range(r/rows):
#                # Conjoin col-space slice and row-space slice (x,y)
#                mask = (slice(0,1), slice(i * rows, (i + 1) * rows), cslice)
#                #print active[mask].shape
#                #print samples.shape
#                if sum(active[mask].shape) >= sum(size):
#                    #print "adding on..."
#                    samples = np.concatenate((samples, active[mask]), axis=0)
#                else:
#                    print active[mask].shape
#                    
#    #print "samples: ", samples.shape
#    samples = gaussian_filter(samples, 1)
#    samples = np.reshape(samples, [len(samples), rows*cols])
#    print "Performing kmeans...."
#    centroids, variance = kmeans(samples, k)
#    centroids = np.reshape(centroids, [len(centroids), rows, cols])
#    print "Save files.."
#    for i in range(len(centroids)):
#        imsave("cen"+str(i)+".jpg", centroids[i])
#    print "Finished"
#    return centroids
    
#out1 = r.hidden_expectation(training_data[i] )
#print out1.shape
#out2 = np.array([r2.hidden_expectation(out1[k]) for k in range(rLayers)])
#print out2.shape
#out1 = np.array([r2.visible_expectation(out2[k]) for k in range(rLayers)])
#print out1.shape
#recon = r.visible_expectation(out1)
#recon = 255*(recon-recon.min())/recon.max()
#imsave('Images\\im'+str(i)+str(j)+'.jpg', recon)

#print 'shape: ', training_data[j].shape
            
#out1 = r.hidden_expectation(test )
#print 'out1 size: ', out1.shape
#out2 = np.array([r2.hidden_sample(out1[k]) for k in range(rLayers)])
#print 'out2 size: ', out2.shape
#out11 = np.array([r2.visible_sample(out2[k]) for k in range(rLayers)])
#print 'out1 size: ', out1.shape
#pool_test = r.pooled_expectation(test)

#print 'pooled out: ', pool_test, ', size: ', pool_test.shape

#for i in range(out1.shape[0]):
##    print 'Filter ', i+1, ':'
#    print 'hidden: '
#    print out1[i]
#    print 'pool: '
#    print pool_test[i]

#recon = r.visible_expectation(out1)

#print 'Reconstruction: ', r.reconstruct(test,2)
#out2 = r2.hidden_expectation( out1 )
#print 'Layer 2: ', out2
#print 'Visible (lay1): ', recon#/float(np.amax(recon))
#print 'Visible (lay2): ', r.visible_expectation(out11) #r.visible_sample(out1)
#print 'Visible(sample):', r.visible_sample(out11)
