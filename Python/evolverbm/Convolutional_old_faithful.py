# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:16:26 2013

@author: Ryan based on Leif's wrong implementation
"""
from RBM import RBM
from utils import *
from PIL import Image
from scipy.misc import imsave
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import *
from MyError import NaNError, MyError
import numpy as np
import logging, threading, pickle
import sys, os, math
import weka, random
import copy, operator
import cProfile
import numpy.random as rng
from datastorage import Data #dl_data
from mnist import MNIST

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
running = True
image_counter =2
def running_func():
    global running
    while True:
        a = raw_input()
        if a == '':
            running = False
learning_rate = .02
oldE, avgE = 90., 10.
oldRE, avgRE, = float("inf"), float("inf")
minRE1, minRE2 = float("inf"), float("inf")
lock = threading.Lock()

class ConvolutionalEvolution(RBM):
    count = 0
    def __init__(self, num_filters, filter_shape, pool_shape, binary=True):
        #print ConvolutionalEvolution.count
        self.id = ConvolutionalEvolution.count
        ConvolutionalEvolution.count += 1
        self.fitness = None
        self.selection_measure = None
        self.genome = []
        self.num_filters = num_filters
        self.weights_size = np.append(num_filters, filter_shape)
        
        for i in range(num_filters):
            wgene = []
            for j in range(10):
                wgene.append( self.generateGene() )
            self.genome.append(wgene)

        self._set_weights()
        
        self.vis_bias = np.zeros(1) #rng.randn(1)   # Scalar visual bias
        
        ''' Bias for each hidden filter'''
        self.hid_bias = np.zeros(num_filters)#rng.randn(num_filters)   # Scalar bias for each filter

        # Identity should be gaussian
        self._visible = binary and sigmoid or identity # gaussian #identity
        #self._visible = binary and sigmoid or gauss
        self._pool_shape = pool_shape
    
    def generateGene(self): # for a filter
        angle = random.randint(0,180)
        return [random.normalvariate(0,.01), random.randint(0,self.weights_size[1]), 
                                           random.randint(0,self.weights_size[2]),
                      math.exp(random.normalvariate(0,1)), math.exp(random.normalvariate(0,1)), angle]
                      
    def _set_weights(self):
        self.weights = np.zeros(self.weights_size)
        for i in range(len(self.genome)):
            for k in range(len(self.genome[i])):
                g = gaussianf(*self.genome[i][k])
                self.weights[i] = self.weights[i] + gaussian_kernel(g, self.weights_size[1])
               
                
    def _pool(self, hidden, down_sample=False):
        ''' Denominator of Equations 14 and 15 (sum elements in Beta)
            Given activity in the hidden units, pool it into groups.
            
            hidden: data to be pooled into groups of size _pool_shape. It is 
                assumed that the data is exp(I(h)) in the paper
            down_sample: determines if resulting image is of original shape or
                proportionally downsampled for each _pool_shape region.
        '''
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
        assert pool.T.shape <= hidden.shape 
        return pool.T
        
    def visible_to_hidden(self,visible, bias=0.):
        ''' Visible to hidden sample'''
        return bernoulli(self.hidden_expectation(visible))
        
    def hidden_expectation(self, visible, bias=0.):
        '''Equation 14 in Lee paper
           Given visible data, return the expected hidden unit values.
        '''
        #For batch weights[k:k+1, ::-1, ::-1]
        if self.weights.max() > 10:
            print str(self.weights.max()),  " ", str(self.weights.min())
        #if self.hid_bias.max() > 10:
        #    print str(self.hid_bias.max()),  " ", str(self.hid_bias.min())
        activation = np.array([
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid') # flipped h & v
            for k in range(self.num_filters)]).T + self.hid_bias + bias
        act = activation.T
        
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
           
    # copy constructor... 
    def copy(self):
        cpcat = copy.deepcopy(self)
        cpcat.id = ConvolutionalEvolution.count
        ConvolutionalEvolution.count += 1
        return cpcat
        
    def calculateFitness(self, data):

        self._set_weights()
        error = self.get_reconstruction_error(data)
        #print "error: ", str(error)
        #if self.fitness is None:
        #    self.fitness = 1000 * (1.+error)/(error+.00000001 + len(self.genome)*1.0/1000000) - error * (10000. + 100. * len(self.genome))
            #100*( 1.0 / (error+1.0) )
        #print "ERROR: ", error
        #self.fitness = 1000. * (1.+error)/(error+.00000001 + len(self.genome)*1.0/1000000) - error * (10000. + 100. * len(self.genome))
        self.fitness = - error
        #100*( 1.0 / (error+1.0) ) - .01 * len(self.genome)

        return self.fitness
       
class EvolveTrainer():
    
    def __init__(self, initPop = 100, children = 100, num_filters=20):
        self.population = []
        self.pop_size = initPop
        #self.children_size = children
        self.avg_fitness = 10
        self.children_pop = []
        self.gauss_mu = .001 # mutator mu
        self.gauss_sigma = 1 # mutator sigma
        for i in range(0,initPop):
            self.population.append(ConvolutionalEvolution(num_filters, [12, 12], [2, 2]))

    def _gaussianMutate(self, number = None):
        
        if number is None:
            number = len(self.population)
        mutated = []
#g = gaussianf(.001, random.randint(0,len(r.weights)-1), random.randint(0,len(r.weights)-1), .001*random.random(), .001*random.random(), random.randint(0, 90))
        #print "begin gaus mutate"

        for p in range (0, number): #len(self.population)):
            p = p % len(self.population)
            #if p == 0:
            #    current = self.population[p]
            #else:
            current = self.population[p].copy()
            for i in range (0, len(self.population[p].genome)): 
                #filter
                
                genefvar = np.var(self.population[p].genome[i][:][0])
                #if i == 0:
                #    print "var: ", genefvar
                #print self.population[p].genome[i]
                #print sum, " ", where(sum)
                #su = sum(self.population[p].genome[i])#, axis=0)
                #mean_scale = su[0] / len(self.population[p].genome[i])
                for g in range (0, len(self.population[p].genome[i])): # each []
                    #for param in range (0, len(self.population[p].genome[i][g]))
                    if random.randint(0,2) == 1:
                        sc = random.normalvariate(0,1) * genefvar / 10000
                        #/mean_scale/1000
                        if sc > 5 or self.population[p].genome[i][g][0] >5:
                            print "WARNING: ", self.population[p].genome[i][g][0], " ",sc
                        #print "mutate: ", sc
                        current.genome[i][g][0] += sc
                        #print "genome: ",current.genome[i][g][0]
                        mx = random.randint(0,5) - 2
                        current.genome[i][g][1] += mx
                        current.genome[i][g][1] %= (current.weights_size[1]-2)
                        if current.genome[i][g][1] < 2:
                            current.genome[i][g][1] += random.randint(0,current.weights_size[1]-3)
                        #if current.genome[i][g][1] > self.population[p].weights_size[1]-2:
                        #    current.genome[i][g][1] = self.population[p].weights_size[1]-2
    
                        my = random.randint(0,5) - 2
                        current.genome[i][g][2] += my
                        current.genome[i][g][2] %= (current.weights_size[2]-2)
                        if current.genome[i][g][2] < 2:
                            current.genome[i][g][2] += random.randint(0,current.weights_size[2]-3)
                        #if current.genome[i][g][2] < 1:
                        #    current.genome[i][g][2] = 2
                        #if current.genome[i][g][2] > self.population[p].weights_size[2]-2:
                        #    current.genome[i][g][2] = self.population[p].weights_size[2]-2
                            
                        va = math.exp(random.normalvariate(0,1) +.1)
                        current.genome[i][g][3] *= va                   
                        #current.genome[i][g][3] %= current.weights_size[1]
                        
                        va = math.exp(random.normalvariate(0,1))
                        current.genome[i][g][4] *= va 
                        #current.genome[i][g][4] %= current.weights_size[2]
                        
                        an = random.normalvariate(0,2) #random.randint(-10,10)
                        current.genome[i][g][5] += an
                        current.genome[i][g][5] %= 180
                        
            current.hid_bias = current.hid_bias + .5 * (rng.randn(self.population[p].num_filters) ) # + current.hid_bias.mean()/10)
            current.vis_bias = current.vis_bias + .5 * rng.randn(1)  
            #if p ==0:
            #    mutated.append(current.copy())
            #else:
            mutated.append(current)
        return mutated
    
    def _insertDeleteMutate(self, number = None):
        
        mutates = []
        if number is None:
            number = len(self.population)
        for i in range (number):
            todel = random.randrange(len(self.population))
            current = self.population[todel].copy()
            filt = random.randrange(len(self.population[todel].genome))
            gene = random.randrange(len(self.population[todel].genome[filt]))
            
            current.genome[filt].pop(gene)
            mutates.append(current)
        
            toadd = random.randrange(len(self.population))
            current = self.population[toadd].copy()
            filt = random.randrange(len(self.population[todel].genome))
            current.genome[filt].append(current.generateGene())
            mutates.append(current)
        return mutates
        
    def _crossover(self, number = None):
        if number is None:
            number = len(self.population)
        mutated = []
        # make sure using right term population THE RBM not pop THE WEIGHTS
        for i in range(number):
            #i = i % len(self.population)
            cross = random.randrange(len(self.population))
            c1 = self.population[cross].copy()
            cross = random.randrange(len(self.population))
            c2 = self.population[cross].copy()
            
            filt_start = random.randrange(len(self.population[cross].genome)//2)
            filt_end = random.randint(len(self.population[cross].genome)//2, 
                                      len(self.population[cross].genome)-1)
            
            c1.genome[filt_end:] = c2.genome[filt_end:]
            c1.genome[:filt_start] = c2.genome[:filt_start]
                    
            c2.genome[filt_start:filt_end] = c1.genome[filt_start:filt_end]
            
            mutated.append(c1)
            mutated.append(c2)

        return mutated

    def setGaussianParams(self, initmu = .001, initsigma = 1):
        self.gauss_mu = initmu
        self.gauss_sigma = initsigma
    def mutate(self, numGaussians=None, numInsertDel=None, numCrossover=None):#, pop):
        # copy parents
        self.children_pop = []
        self.children_pop.extend(self._gaussianMutate(numGaussians))
        #self.children_pop.extend(self._gaussianMutate())
        self.children_pop.extend(self._insertDeleteMutate(numInsertDel))
        self.children_pop.extend(self._crossover(numCrossover))
        #self.children_pop.extend(self._scaleMutate())
        # mutate
        pass

    def nextgen(self, data):
        # combine parents and children
        self.population.extend(self.children_pop)
        
        for i in xrange(len(self.population)):
            self.avg_fitness = .99*self.avg_fitness + .01*self.population[i].calculateFitness(data)
            self.population[i].selection_measure = self.population[i].calculateFitness(data) /abs(self.avg_fitness) - random.normalvariate(0,1)
            #print self.population[i].fitness

        #first = self.population[0]
        # set new population
        #bid = self.population[5].id
        #self.population.
        self.population.sort(key=lambda x: x.selection_measure, reverse=True)
        
        #assert self.population[5].id != bid
        #self.population[self.pop_size-1] = self.population[0]
        #self.population[0] = first # see consistency
        self.population = self.population[0:self.pop_size]
        self.pop_size = len(self.population)
        pass
    def learn(self, visible ):#, learning_rate=0.2, cdk=1):

        pass
        self.mutate()#self.population)
        self.nextgen(visible)
    def visualizePop(self, todo = 'show', popSelect = None, filterSelect = None, saveType = None):
        #plt.hold(False)
        if popSelect is None:
            popSize = len(self.population) #min (len(self.population), 30)
        else:
            popSize = len(popSelect)
        if filterSelect is None:
            numfilters = len(self.population[0].genome)
        else:
            numfilters = len(filterSelect)
        fig = plt.figure(1, (80., 80.))
#        fig.clf()
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                        nrows_ncols = (numfilters, popSize), # creates 2x2 grid of axes
                        axes_pad=0.1, # pad between axes in inch.
                        )
        #print "vize: ", grid.get_vsize_hsize
#        print "Size of grid: ", sys.getsizeof(grid)
#        print "Size of fig: ", sys.getsizeof(fig)
        #for i in range(numfilters):
        #    for j in range(popSize):
        #        grid[i*popSize+j].imshow(self.population[j].weights[i])
        if popSelect is None and filterSelect is None:
            for i in range(numfilters):
                for j in range(popSize):
                    grid[i*popSize+j].imshow(self.population[j].weights[i]) # The AxesGrid object work as a list of axes.
        elif filterSelect is None:
            for i in range(numfilters):
                for j in popSelect:
                    grid[i*popSize+j].imshow(self.population[j].weights[i])
        else:
            for i in filterSelect:
                for j in popSelect:
                    grid[i*popSize+j].imshow(self.population[j].weights[i])
        if todo == 'show':
            plt.show()
        elif todo == 'save':
            if saveType is None:
                fig.savefig("epoch_" + "_" + "_avfit_" + str(abs(int(self.avg_fitness)))+ "_pop.png")
            else:
                fig.savefig("epoch_" + "_" + "_avfit_" + str(abs(int(self.avg_fitness)))+ "_pop_"+saveType+".png")
            #str(self.data_loc) + 
        print "Filename: ", "epoch_" + "_" + "_avfit_" + str(abs(int(self.avg_fitness)))+ "_pop.png"

        plt.cla()
        plt.hold(False)
        #plt.close(fig)
        #plt.close("all")
        #gc.collect()

def testEvolution():
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
    training_data = (training_data * 1.0)/255 #training_data.max()
    #imsave("Images\\reconL0_"+str(20)+".jpg", training_data[0])
    saveType = "genone" #20
    np.random.shuffle(training_data)
    
    training_data = training_data[0:50000]
    print 'Test ConvRBM'
    rLayers = 5
    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
    #r = ConvolutionalEvolution(rLayers, [15,15], [2,2], False) #ConvolutionalEvolution(2, [3, 3], [2, 2])
    t = EvolveTrainer(40, 400, rLayers)
    
    #t = pickle.load(open("trainer.p", "rb"))#
    #for i in range(len(t.population)):
    #    r = pickle.load(open("conv"+str(i)+".p", "rb"))
    #    t.population[i] = r
    t.visualizePop('save',saveType = "genone")
    #return 0
    #for i in range(len(t.population)):
    #    t.population[i].fitness = -9999
    r = t.population[0]
    batchsize = 15
    
    '''Trainer(rbm, momentum=0., l2=0., target_sparsity=None):'''
    #t = ConvolutionalTrainer(r,.5, 0, .005) #changed from .005 to .05
    print 'Training...'
    for i in range(rLayers):
        imsave(os.path.join("Images", "weights_init_" +saveType+"_"+str(i)+".jpg"), r.weights[i])
    ''' Training for first layer'''
    for i in range(10000):
            ''' Get NEW training data'''
            #        def trainThread():
            global avgRE, learning_rate, minRE1
            np.random.shuffle(training_data)
            for j in range(0, training_data.shape[0], batchsize):
                #oldRE = avgRE
                
                print "Looping..."

                print "bat size: ", training_data[j:j+batchsize].shape
                t.learn(training_data[j:j+batchsize])
                #avgRE = r.get_avg_error(training_data[j])
                print "num children: ", len(t.children_pop)
                print "pop size: ", t.pop_size
                print "avg fitness: ", t.avg_fitness
                print ".... Success!"
                r = t.population[0]
                print "id: ", str(r.id), " averages: (w, h, v) ", str(r.weights.mean()), ", ", str(r.hid_bias.mean()), ", ", str(r.vis_bias)
                
                #print "id next: ", str(t.population[0].id)
                print "r avg value: ", r.weights.mean()
                for k in range(rLayers):
                    imsave(os.path.join("Images", "weights_iter"+str(i)+saveType+str(k)+".jpg"), r.weights[k])
                #    break
            
            
                if j % 1 == 0 :
                    ''' Reconstruct image for one layer'''                    
                    minRecon = r.reconstruct(training_data[j], 2)
                    #minRecon = minRecon / minRecon.max() * 255
                    with lock:
                        imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_0.jpg"), training_data[j]*255)
                        imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_1.jpg"), minRecon*255)
                    t.visualizePop('save',saveType = "genone")
                if not running:
                    with lock:
                        print 'First break'
                        print 'Breaking on running (in)'
                    break

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
    
    for i in range(len(t.population)):
        r = t.population[i]
        pickle.dump(r, open("conv"+str(i)+".p", "wb"))
    print 'Saving layer 1 weights'
    pickle.dump(t, open("trainer.p", "wb"))
    
    # Print weights to images
    #for i in range(rLayers):
    #    imsave(os.path.join("Images", "weightsL20_"+str(i)+".jpg"), r.weights[i])
    #thr.join()
    #print 'joined.'
    print 'Done.'
    
if __name__ == '__main__':
    #cProfile.run('testConvolution()')
    #testConvolution()
    testEvolution()
    
   