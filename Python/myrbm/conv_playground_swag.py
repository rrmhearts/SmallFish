# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:16:26 2013

@author: Ryan based on another implementation
"""
from RBMPlay import RBM, Trainer
from utils import *
from PIL import Image
from scipy.misc import imsave
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import *
#from MyError import NaNError, MyError
from operator import add, div
import numpy as np
import logging
import threading
import pickle
import sys, os, time
import weka
import random, math
import cProfile
import numpy.random as rng
from datastorage import Data #dl_data

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import csv

running = True
image_counter =30
def running_func():
    global running
    while True:
        a = raw_input()
        if a == '':
            running = False
learning_rate = 30 #2 # .1
oldE, avgE = 90., 10.
oldRE, avgRE, = float("inf"), float("inf")
minRE1, minRE2 = float("inf"), float("inf")
lock = threading.Lock()
def getData(loc = ["C:\\Users\\Ryan\\Documents\\SWAG_SUBSET\\male\\Zoom", "C:\\Users\\Ryan\\Documents\\SWAG_SUBSET\\female\\Zoom"], title="SWAG_data"):
    ''' Get data from location, returns normalized data to [0,1] and the respective classes
    '''
    d = Data(loc)
    input_data = d.get_people_set()
    #print input_data[0][0].shape
    #print input_data[0][1]
    #print input_data[0][1][1]
    
    ''' Remove images that are not of the right size'''
    for subject in input_data:
        pic = subject[0]#input_data[i][0]
        if pic.shape != (150,90,3):
            print 'Removed an element'
            input_data.remove(subject)#input_data[i]) #this broke?alueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            print 'New size: ', len(input_data)

    ''' Pull correct images into data array'''
    data = np.zeros((len(input_data), input_data[0][0].shape[0],input_data[0][0].shape[1]))
    for i in range(len(input_data)):
        pic = input_data[i][0]
        #print type(pic), ' ', pic.shape
        if pic.shape == (150,90,3):
            #print 'success!'
            data[i,:,:] = Image.fromarray(pic, 'RGB').convert('L')
        else:
            raise MyError(pic.shape)
            #data[i,:,:] = data[i-1,:,:]
    ''' Get classes from data'''
    classes = [subject[1] for subject in input_data]
    #print classes
    pickle.dump([data, classes], open(title+".p", "wb"))
    return data, classes

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
        self.weights = scale * (rng.randn(num_filters, *filter_shape))
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
        self.hidexp = None #np.ones(filter_shape)

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
                / (self._pool_shape[0]*self._pool_shape[1]) -1):
            #  Unstable 1 - P(pool=0 | v)
            pool0 = 1. / (1. + self._pool(np.exp(act), True))
        else: #except OverflowError:
            act = act / act.max() * np.log(np.finfo(dtype).max 
                / (self._pool_shape[0]*self._pool_shape[1]) -1)
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
        
        #act = prevent_overflow(act)
        #act[np.isnan(act)] = 0
        act = np.exp(act)
        expectation = act / (1. + self._pool(act) )
        
        #if self.hidexp is None:
        #    self.hidexp = expectation.mean(axis=0)
        #    self.hidexp = self.hidexp/self.hidexp.max()
        #else:
        #    self.hidexp = .99*self.hidexp + .01*expectation.mean(axis=0)/expectation.mean(axis=0).max()
        #expectation = np.array([ (expectation[i] + expectation[(i+1)%len(act)] ) % 2 for i in range(len(act)) ])
        #assert expectation.max() == 1
        #expmax = np.max(expectation, axis = 0)
        #print expmax.shape
        #expectation[expectation < expmax-.3] = 0
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

    def visualizeRecons(self, data):
        numRecons = len(data)
        fig = plt.figure(1, (30., 60.))
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                        nrows_ncols = (2, numRecons), # creates 2x2 grid of axes
                        axes_pad=0.1, # pad between axes in inch.
                        )
        
        for i in range(numRecons):
            grid[i].imshow(data[i])
            grid[i+numRecons].imshow(self.reconstruct(data[i], 2)) # The AxesGrid object work as a list of axes.
        #fig.savefig("Hello.png")

        plt.show()
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
        #print gradients[0].shape
        #if fi < 50:
        #for i in range(len(gradients[0])):
        #imsave(os.path.join("Images", "gradient__0.jpg"), gradients[0][0])
        with lock:
            self.apply_gradients(*gradients, learning_rate=learning_rate)
        return np.abs(gradients[0]).mean()
    def learn(self, rbm, visible, learning_rate=0.2, cdk=1):
        ''' Calculate and apply gradients
            visible: data to learn
            learning_rate: How much gradient affects the weights
            cdk: How far to take contrastive divergece (1 is quick)
        '''
        self.rbm = rbm #.copy()
        #self.learning_rate = 0.99*self.learning_rate + 0.01*learning_rate
        gradients = self.calculate_gradients(visible,cdk)
        #print gradients[0].shape
        #if fi < 50:
        #for i in range(len(gradients[0])):
        #imsave(os.path.join("Images", "gradient__0.jpg"), gradients[0][0])
        with lock:
            self.apply_gradients(*gradients, learning_rate=learning_rate)
        return np.abs(gradients[0]).mean()
        
    def calculate_gradients(self, visible, cdk=1):
        ''' Calculate gradients for an instance of visible data.
            Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

            visible: A single array of visible data.
        '''
        
        ''' Get hidden representations (h0,h1) and reconstructed visible (v1)'''
        passes = self.rbm.iter_passes(visible)
        v0, h0 = passes.next()
        #h0max = np.max(h0, axis = 0)
        #print expmax.shape
        #h0[h0 < h0max-.1] = 0
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
        if not self.upper:        
            gv = (v0 - v1).sum() /v0.size #/10
        else:
            gv = 0
            
        ''' Equation 18. One bias for each hidden filter'''
        gh = (h0 - h1).sum(axis=-1).sum(axis=-1)*1.0/h0.size #/h0.size
 
        ''' Equation 16'''
        if self.target_sparsity is not None:
            self.sparseness = self.target_sparsity - self.rbm.hidden_expectation(visible).mean(axis=-1).mean(axis=-1)
            #self.rbm.hidden_sample(visible).mean(axis=-1).mean(axis=-1)
            #self.rbm.hidden_expectation(visible).mean(axis=-1).mean(axis=-1)
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
        #print "gradient average: ", gw.mean()
        #gw = 2*(gw - gw.mean(axis=1).mean(axis=1).reshape(len(gw),1,1)) + gw.mean(axis=0) #*gw.mean()
        #gw = 2*gw - gw.mean(axis=1).mean(axis=1).reshape(len(gw),1,1)  #) + gw.mean(axis=0) #*gw.mean()
        #gw = np.array([ gw[i] *(gaussian_kernel (gaussianf(1, random.uniform(max( (i%3)*4-1,0), min( (i%3+1)*4, gw.shape[1])), # mean, x,
        #            random.uniform(max(int(i/7)*4,0), min( int(i/7+1)*4, gw.shape[1])), abs(random.normalvariate(1,1)),           # y, stdx,
        #            abs(random.normalvariate(1,1)), random.uniform(0,180) ), gw.shape[1] ) +.5) for i in range(len(gw)) ]) #stdy, angle), size
        #print gw[0]
        
        #print gaussian_kernel (gaussianf(1, random.uniform(0,5), random.uniform(0,5), 
        #              abs(random.normalvariate(1,1)), abs(random.normalvariate(1,1)), random.uniform(0,180) ), 5 )+.5
                      
        #print gw.shape
        return gw, gv, gh


def testConvolution():
    ''' Second thread used to break out of training loop'''
    thr = threading.Thread(target=running_func)
    thr.start()
    global running
    
    tort = "TRAINING"
    training_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\female\\Zoom\\F"])

    print training_data.shape
    
    saveType = "SWAG" #20
    #np.random.shuffle(training_data)
    #training_data = training_data[0:10000]# 1000]
    
    np.random.seed(101)
    np.random.shuffle(training_data)
    training_data = training_data[0:10000]*1.0/255
    
    #training_data = training_data[0:2000]
    print 'Test ConvRBM'
    rLayers = 20
    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
    r = Convolutional(rLayers, [12,12], [2,2], True, .1) #Convolutional(2, [3, 3], [2, 2])

#    r = pickle.load(open("minConvLayer"+saveType+".p", "rb"))#
    #r = pickle.load(open("convLayer.p", "rb"))#
    #r.vis_bias = r.vis_bias -30
    #r.hid_bias = r.hid_bias -1
    #r = pickle.load(open("convLayer"+saveType+".p", "rb"))#
    #r._pool_shape = [3,3]
    #r.num_filters = 20
    #print r.weights.shape
    #print r.weights[0:20].shape
    #temp = r.weights + .05*np.random.rand(20,7,7)
    #r.weights = temp
    #r.error = 150
    #r.weights[5] = r.weights[6,:,::-1]
    #r.weights[14] = r.weights[14].T
    #r.hidexp = np.ones((17,17))
    '''Trainer(rbm, momentum=0., l2=0., target_sparsity=None):'''
    t = ConvolutionalTrainer(r,.5, 0.0001, .05) #.00001, .005) #changed from .005 to .05
#    t = pickle.load(open("minTrainer"+saveType+".p", "rb"))#
    #t = ConvolutionalTrainer(r,0, 0, 0)
    rLayers = r.num_filters
    #rate = [3, 8, 10]
    gradients = t.calculate_gradients(training_data[0],2)
    #for i in range (10):
    #    gradients = tuple(map(add, gradients, t.calculate_gradients(training_data[min(j+i,len(training_data)-1)],2) ) )
    batchsize = 15
    #rbms = [r.copy(), r.copy(), r.copy()]
    print 'Training...'
    for i in range(rLayers):
        imsave(os.path.join("Images", "weightsLinit" +saveType+"_"+str(i)+".jpg"), r.weights[i])
    ''' Training for first layer'''
    
    #r.visualizeRecons(training_data[0:20])
    firstfull = False
    secondfull = False
    thirdfull = False
    
    start = time.clock()
    with open('SWAG_History.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(t.stats) 
    t.stats = []
    for i in range(t.epoch, 25000):
            t.epoch = i
            ''' Get NEW training data'''
            #        def trainThread():
            global avgRE, learning_rate, minRE1
            np.random.shuffle(training_data)
            for j in range(0, training_data.shape[0], batchsize):
                oldRE = avgRE
                
                #if i < 50 and i %1 == 0:
                    #minRecon = r.reconstruct(training_data[j], 2)
                    #imsave(os.path.join("Images", "reconA"+saveType+"_"+str(i)+"_1.jpg"), minRecon)
                    #for k in range(rLayers):
                    #    imsave(os.path.join("Images", "weightsA"+saveType+"_min_"+str(i)+"_k"+str(k)+".jpg"), r.weights[k])
                    
                ''' Slowly decrease learning rate from ~1/500th of the mean weight'''
                #learning_rate = .99*learning_rate + .01*(abs(float(r.weights.mean())) )/(10 )#+ i*i) )
                #learning_rate = 10
                
                #self.learning_rate = 0.99*self.learning_rate + 0.01*learning_rate
                #lning = t.learn(cr, training_data[j], rate, cdk=2)
                
                #besti = 1
                #gradients = tuple(x/1000 for x in gradients)
                for k in range (batchsize):
                    gradients = tuple(map(add, gradients, t.calculate_gradients(training_data[min(j+k,len(training_data)-1)],2) ) )
                gradients = tuple(map(div, gradients, (batchsize,)*3))

                t.apply_gradients(*gradients, learning_rate=learning_rate)

                #lning = t.learn(training_data[j], learning_rate, cdk=2)
                avgRE = r.get_avg_error(training_data[j:min(j+batchsize, len(training_data)-1)])
                #avgRE = r.get_reconstruction_error(training_data[j:min(j+batchsize, len(training_data)-1)])
                elapsed = (time.clock() - start)
                
                #t.stats.append((i, j, learning_rate, avgRE, elapsed))
                
                print r.weights.mean(), " ", r.hid_bias.mean(), r.vis_bias.mean()
                # If error stops decreasing over 100 iterations, break loop
                if j+i*(training_data.shape[0]) % 9999 == 1:
                    oldRE = avgRE
                
                
                ''' Save minimum weights'''
                if avgRE < oldRE:
                    direction = '-'
                    
                    if avgRE < minRE1:
                        minRE1 = avgRE
                        #print r.hidexp
                        if (i+j+random.randint(0, 9)) % 10 == 1 :
                            ''' Reconstruct image for one layer'''                    
                            minRecon = r.reconstruct(training_data[j], 2)
                            #minRecon = minRecon / minRecon.max() * 255
                            with lock:
                                imsave(os.path.join("Images", "reconL"+saveType+"__0.jpg"), training_data[j])
                                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i%3)+"_1.jpg"), minRecon)
                                #+str(i*100+j)+"_1.jpg"), minRecon)
                        f = open('minstats.txt', 'w')
                        f.write('Epoch: '+str(i)+' Image: '+str(j)+' BS: '+str(batchsize)+' lr: '+str(learning_rate)+' Time: '+str(elapsed)+'\n')
                        f.close()
                        if (i+j+random.randint(0, 9)) % 10 == 1 : #minRE1 < 2000 and
                            with lock:
                                print 'Saving...'
                                pickle.dump(r, open("minConvLayer"+saveType+".p", "wb"))
                                pickle.dump(t, open("minTrainer"+saveType+".p", "wb"))
                                for k in range(rLayers):
                                    imsave(os.path.join("Images", "weightsL"+saveType+"_min_"+str(k)+".jpg"), r.weights[k])
    

                        #if avgRE < 20 and firstfull is not True:
                        #    r.visualizeRecons(training_data[0:20], 'save')
                        #    firstfull = True
                        #elif avgRE < 30 and secondfull is not True:
                        #    r.visualizeRecons(training_data[0:20], 'save')
                        #    secondfull = True
                        #elif avgRE < 50 and thirdfull is not True:
                        #    r.visualizeRecons(training_data[0:20], 'save')
                        #    thirdfull = True
                            
                    if abs(oldRE - avgRE) < .01: 
                        t.momentum = .7
                    #learning_rate = .999*learning_rate + .1
                else:
                    direction = '+'
                    learning_rate = .995*learning_rate + .005*(abs(float(r.weights.mean())) )/(10000 )#+ i*i) )
                    t.l2 = .995*t.l2 + .005*t.l2/2

                with lock:
                    print i, 'Error: ', avgRE, direction, 'grad avg: ', gradients[0].mean()
                    
                    print "Time elapsed: ", elapsed, " Learning rate: ", str(learning_rate)
                #if abs(oldRE - avgRE) < .01:
                #    break
                if not running:
                    with lock:
                        print 'First break'
                        print 'Breaking on running (in)'
                    break

            if not running:
                print 'Second break'
                print 'Breaking on running (out)'
                break
                #print 'shape: ', r.hidden_sample(training_data[j]).shape
    #with lock:
    #    print 'Joining threads...'
    #thr_train.join()
    print 'Saving layer 1 weights'
    pickle.dump(r, open("convLayer"+saveType+".p", "wb"))
    pickle.dump(t, open("trainer"+saveType+".p", "wb"))
    minRecon = r.reconstruct(training_data[j], 2)
    with lock:
        imsave(os.path.join("Images", "reconL"+saveType+"__0.jpg"), training_data[j])
        imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i%3)+"_1.jpg"), minRecon)
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
    tort = "TRAINING"
    training_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\female\\Zoom\\F"])

    print training_data.shape
    
    saveType = "SWAG" #20
    
    np.random.seed(103)
    np.random.shuffle(training_data)
    np.random.shuffle(classes)
    data = training_data#[0:1000]*1.0/255
    #classes = classes[0:1000]
    r = pickle.load(open("minConvLayer"+saveType+".p", "rb"))#
    t = pickle.load(open("minTrainer"+saveType+".p", "rb"))#
    amt_training = 1000
    tort = "TRAINING"
    # Change to numpy array
    #classes = np.fromiter(classes, dtype=np.int)
    print len(data)
    assert len(data) == len(classes)    
    #p = np.random.permutation(len(data))
    #data, classes = data[p], classes[p]
    
    #testing_data = data[0:1000]
    #testing_classes = classes[0:1000]
    
    #training_data = data[1000:1000+amt_training]
    #training_classes = classes[1000:1000+amt_training]
    training_data = data#[0:1000+amt_training]
    training_classes = classes#[0:1000+amt_training]
    
    weka_file = tort+"_SWAG_output.arff"
    wk = weka.Weka(relation="Number")
    last_percent, flag = -1, True
#
    print 'Writing Weka file: ', weka_file
    for i in range(len(training_classes)):
        out1 = r.pooled_expectation(training_data[i] )
        #print 'out1 size: ', out1.shape
        #features = np.array([r2.pooled_expectation(out1[k]) for k in range(rLayers)])
        features = out1
        #features = np.append(out1, features)
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
    wk.add_attribute('class', "{male, female}")
    wk.write_file(weka_file)
    print 'Finished!'
    
    
    
##         def updateHistory(self):
##         try:
##             self.weights_history = np.concatenate((self.weights_history, 
##                         np.expand_dims(self.weights, axis=0)), axis=0)
##         except AttributeError:
##             self.weights_history = np.expand_dims(self.weights, axis=0)
##         
##     def writeHistory(self, i):
##         try:
##             pickle.dump(self, open("backupNetwork.p", "wb"))
##             pickle.dump(self.weights_history, open("weightsHistory_"+str(i)+".p", "wb"))
##         except MemoryError:
##             print "Memory error!"
##         except IOError:
##             print "He's dead, Jim!"
##             pickle.dump(self, open("backupNetwork.p", "wb"))