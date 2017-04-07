# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:16:26 2013

Rizki Playground

@author: Ryan based on Leif's wrong implementation
"""
from RBM import RBM
from utils import *
from PIL import Image
from scipy.misc import imsave
from scipy.ndimage import gaussian_filter
from scipy.cluster.vq import *
import numpy as np
import logging, threading, pickle
import sys, os, math, time, gc
import weka, random
import copy, operator
import cProfile
import numpy.random as rng
from mnist import MNIST
from operator import attrgetter

import cv2      # RIZKI


import matplotlib as mpl
#print mpl.get_backend()
#mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure

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
lock = threading.Lock()

class ConvolutionalEvolution(RBM):
    ''' Static count variable for the id '''
    count = 0
    def __init__(self, num_filters, filter_shape, pool_shape, binary=True, num_gaussians=5):
        '''
            num_filters: Number of filters for the CRBM
            filter_shape: weight shape [x,y]. usually x=y
            pool_shape: pooling region. 
            binary: whether the input data is binary or gaussian. Effects visible expectation
            num_gaussians: number of gaussians to init in each filters
        '''
        self.id = ConvolutionalEvolution.count
        ConvolutionalEvolution.count += 1
        self.fitness = -9999
        self.selection_measure = 0
        self.popularity = 0
        self.genome = []
        self.filter_changed = np.ones(num_filters, dtype=bool)
        self.num_filters = num_filters
        self.weights_size = np.append(num_filters, filter_shape)
        self.weights = np.zeros(self.weights_size)
        self.error = -1
        
        ''' Initialize genome'''
        for i in range(num_filters):
            wgene = []
            for j in range(num_gaussians):
                wgene.append( self.generateGene() )
            self.genome.append(wgene)
            
        ''' Set weights according to genome'''
        self._set_weights()
        
        ''' Visible bias'''
        self.vis_bias = np.zeros(1) #rng.randn(1)   # Scalar visual bias
        
        ''' Bias for each hidden filter'''
        self.hid_bias = np.zeros(num_filters)#rng.randn(num_filters)   # Scalar bias for each filter

        ''' Sigmoid or gaussian for input type'''
        self._visible = binary and sigmoid or identity # gaussian #identity
        #self._visible = binary and sigmoid or gauss
        self._pool_shape = pool_shape
    
    def generateGene(self): # for a filter
        ''' Generate a gene consisting of random initialized values'''
        angle = random.randint(0,180)
        return [abs(random.normalvariate(0.01,.001)), random.randint(4,self.weights_size[1]-4), 
                                           random.randint(4,self.weights_size[2]-4),
                      math.exp(random.normalvariate(0.1,1)), math.exp(random.normalvariate(0,1)), angle]
                      
    def _set_weights(self):
        ''' Set weights according to genome. Map genome to weight matrices
            Only upddates fiters that have corresponding changes in the genome.
        '''
        for i in range(len(self.genome)):
            if self.filter_changed[i]:
                self.weights[i] = 0
                for k in range(len(self.genome[i])):
                    g = gaussianf(*self.genome[i][k])
                    self.weights[i] = self.weights[i] + gaussian_kernel(g, self.weights_size[1])

        # RIZKI
        # display the weights (scale between 0-1 so imshow will display these as ((w-min(w))/(max(w)-min(w))) * 255) as an image, also dump the values to the console
        for i in range(len(self.weights)):
            w = 100.0 * self.weights[i]  # multiplied by big number to spread the values for display purposes
            cv2.imshow('image', (w - np.min(w) / (np.max(w) - np.min(w))) )
            print self.weights[i]
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.filter_changed = np.zeros(len(self.filter_changed), dtype=bool)
                
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
           
    def copy(self):
        '''copy constructor '''
        cpcat = copy.deepcopy(self)
        cpcat.id = ConvolutionalEvolution.count
        ConvolutionalEvolution.count += 1
        return cpcat
        
    def calculateFitness(self, data):
        ''' Fitness function for machine based on the negative reconstruction error.'''

        # Set-up Weights to reflect the genome
        self._set_weights()
        
        #print "Reconstruction.."
        error = self.get_reconstruction_error(data)
        #self.fitness = 1000. * (1.+error)/(error+.00000001 + len(self.genome)*1.0/1000000) - error * (10000. + 100. * len(self.genome))
        numgaussians = 0
        for i in self.genome:
            numgaussians += len(i)
        self.fitness = - error #- error*numgaussians/10000000
        # self.fitness = 100.0 - error
        #100*( 1.0 / (error+1.0) ) - .01 * len(self.genome)

        return self.fitness
       
class EvolveTrainer():
    
    def __init__(self, initPop = 100, children_mult = 4, num_filters=20, tourney=5, filter_size=12, num_gaussians=5, mutation_prob = .7):
        '''
            Evolve trainer constructor. Train on a population of CRBM genomes.
            initPop: size of the population throughout
            children_mult: number children in terms of a multiple of pop_size
            num_filters: Number of filters for each CRBM. Impacts genome size
            tourney: Number of rounds for tournament selection measure.
            filter_size: Size of each convolutional filter on one side [filter_size, filtersize]
            num_gaussians: Number of gaussians used in the genome for each filter
        '''
        self.population = []
        self.pop_size = initPop
        self.tournament_size = tourney
        self.mutationProbability = mutation_prob
        #self.children_size = children
        self.children_multiplier = children_mult
        self.avg_fitness = -200
        self.best_fitness = 100
        self.worst_fitness = -200
        self.min_reconerror = 1000
        self.children_pop = []
        self.best_hid_bias = np.zeros(num_filters)
        self.best_vis_bias = np.zeros(1)
        self.gauss_mu = .001 # mutator mu
        self.gauss_sigma = 1 # mutator sigma
        self.epoch = 0
        self.data_loc = 0
        self.stats = []
        
        ''' Set up initial population'''
        for i in range(0, self.pop_size):
            self.population.append(ConvolutionalEvolution(num_filters, [filter_size, filter_size], [2, 2], num_gaussians=num_gaussians))

    def getNumberMutations(self, probability = .5):
        ''' Randomly choose number of mutations for a child. 
            Based on the geometric distribution
        '''
        count = 0
        while (random.uniform(0,1) < probability):
            count += 1
        return count
        
    def _gaussianMutate(self, aChild, number ): #gmutateprob=.7): #number = None):
        
        #if number is None:
        #number = self.getNumberMutations(gmutateprob)

        #print "Number mutations: ", str(number)

        for j in range (0, number): #len(self.population)):
            # filter
            i = random.randrange(len(aChild.genome))
                
            g = random.randrange(len(aChild.genome[i]))
            aChild.filter_changed[i] = True
            if random.randint(0,4) == 1:
                if random.randint(0,1) == 1:
                    scale_den = abs(1+aChild.id * random.normalvariate(0,1)/aChild.genome[i][g][0])*10
                else:
                    scale_den = abs(1+100000 * random.normalvariate(10,2)/aChild.genome[i][g][0])
                    
                sc = random.normalvariate(.05,1) / scale_den
                #/mean_scale/1000
                if sc > 5 or aChild.genome[i][g][0] >5:
                    print "WARNING: ", aChild.genome[i][g][0], " ",sc
                    sc = -sc#/10
                #print "mutate: ", sc
                aChild.genome[i][g][0] += sc
            if random.randint(0,4) == 1:
                mx = random.randint(0,5) - 2
                if aChild.genome[i][g][1]+mx < 3 or \
                    aChild.genome[i][g][1]+mx > aChild.weights_size[1]-4:
                    mx = -mx
                aChild.genome[i][g][1] += mx
            if random.randint(0,4) == 1:
                my = random.randint(0,5) - 2
                if aChild.genome[i][g][2]+my< 3 or \
                    aChild.genome[i][g][2]+my > aChild.weights_size[2]-4:
                    my = -my
                aChild.genome[i][g][2] += my
                
            if random.randint(0,5) == 1:
                
                offset = 0
                total_var = aChild.genome[i][g][3] + aChild.genome[i][g][4]
                if total_var < .5:
                    offset += (1-total_var)
                
                va = math.exp(random.normalvariate(offset+.2,.1))
                aChild.genome[i][g][3] *= va                   
                #current.genome[i][g][3] %= current.weights_size[1]
            #if random.randint(0,3) == 1:
            elif random.randint(0,4) == 1:
                offset = 0
                total_var = aChild.genome[i][g][3] + aChild.genome[i][g][4]
                if total_var < .5:
                    offset += (1-total_var)
                    
                va = math.exp(random.normalvariate(offset+0,.1))
                aChild.genome[i][g][4] *= va 
                
                #current.genome[i][g][4] %= current.weights_size[2]
            elif random.randint(0,3) == 1:
                # Grab another gene from filter                
                g_off = random.randrange(len(aChild.genome[i]))
                # Encourage a common angle within a filter
                g_angle = aChild.genome[i][g_off][5]
                devotion = aChild.genome[i][g_off][3]
                #nearness_45 = (aChild.genome[i][g_off][5] % 90) / 10
                an = random.normalvariate(0,5) #random.randint(-10,10)
                aChild.genome[i][g][5] += (an + .01*devotion*(g_angle - aChild.genome[i][g][5]) )
                aChild.genome[i][g][5] %= 180
            #if random.randint(0,3) != 1:           
            # Always move the bias terms.. there is one best.
            if random.randint(0,1) == 1:
                aChild.hid_bias = .6*aChild.hid_bias + rng.randn(aChild.num_filters)\
                                 + .4*self.best_hid_bias
                aChild.vis_bias = .6*aChild.vis_bias + rng.randn(1) + .4*self.best_vis_bias

    
    def _insertDeleteMutate(self, aChild, numberInserts, numberDeletes):
        
        #mutates = []
        #if number is None:
         #   number = len(self.population)
        #numberInserts = self.getNumberMutations(insertmutateprob)
        #numberDeletes = self.getNumberMutations(deletemutatateprob)
        
        #print "Number IDs, ins: ", str(numberInserts), " dels: ", str(numberDeletes)

        for i in range (numberInserts):
            #toadd = random.randrange(len(self.population))
            #current = self.population[toadd].copy()
            filt = random.randrange(len(aChild.genome))
            
            aChild.filter_changed[filt] = True
            aChild.genome[filt].append(aChild.generateGene())
            #mutates.append(current)
        for i in range (numberDeletes):
            #todel = random.randrange(len(self.population))
            #current = self.population[todel].copy()
            filt = random.randrange(len(aChild.genome))
            
            if len(aChild.genome[filt]) > 1 :
                gene = random.randrange(len(aChild.genome[filt]))
            
                aChild.filter_changed[filt] = True
                aChild.genome[filt].pop(gene)
            else:
                # Emergency add gene
                aChild.genome[filt].append(aChild.generateGene())
            #mutates.append(current)
        #return mutates
        
    def _crossover(self, aChild, number):
        #if number is None:
        #    number = len(self.population)
        #mutated = []
        #number = cmutateNumber #self.getNumberMutations(cmutateprob)
        
        #print "Number crossover: ", str(number)
        # make sure using right term population THE RBM not pop THE WEIGHTS
        for i in range(number):
            #i = i % len(self.population)
            cross = random.randrange(len(self.population))
            c1 = self.population[cross].copy()
            #cross = random.randrange(len(self.population))
            #c2 = self.population[cross].copy()
            
            filt_start = random.randrange(len(self.population[cross].genome)-2)
            filt_end = random.randint(filt_start+1, 
                                      len(self.population[cross].genome)-1)
            
            #c1.genome[filt_end:] = c2.genome[filt_end:]
            #c1.genome[:filt_start] = c2.genome[:filt_start]
                    
            aChild.genome[filt_start:filt_end] = c1.genome[filt_start:filt_end]
            
            #c1.filter_changed[filt_end:] = True
            #c1.filter_changed[:filt_start] = True
            aChild.filter_changed[filt_start:filt_end] = True
            
    def _scaleMutate(self, aChild, number):
        
#        if random.uniform(0,1) < scalemutatateprob:
        for i in range(number):
            shift = random.normalvariate(1,.00001*random.uniform(0,50))
            for filt in range(len(aChild.genome)):
                aChild.filter_changed[filt] = True
                for g in range(len(aChild.genome[filt])):
                    aChild.genome[filt][g][0] *= shift
#        elif random.uniform(0,1) < scalemutatateprob:
#            shift = random.normalvariate(0,.01)
#            aChild.hid_bias = aChild.hid_bias + shift
#        elif random.uniform(0,1) < scalemutatateprob:
#            shift = random.normalvariate(0,.01)
#            aChild.vis_bias = aChild.vis_bias + shift
            
    def mutate(self, numberChildren, gmutateprob=None, INSmutateprob=None, DELmutateprob=None, cmutateprob=None, scalemutatateprob=None):#, pop):
        print "Mutating..."
        if scalemutatateprob is None:
            scalemutatateprob = abs(1. - gmutateprob - INSmutateprob - DELmutateprob - cmutateprob)
        totalProb = gmutateprob + INSmutateprob + DELmutateprob + cmutateprob + scalemutatateprob
        if totalProb > 1:
            gmutateprob /= totalProb
            INSmutateprob /= totalProb
            DELmutateprob /= totalProb
            cmutateprob /= totalProb
            scalemutatateprob /= totalProb

        self.children_pop = []
        #self.tournament_size = 5
        
        i = 0
        second = False
        used = []
        while (len(self.children_pop) < numberChildren):
            numberMutations = 1 + self.getNumberMutations(self.mutationProbability) #.8)
        
            gmutateNumber, INSmutateNumber, DELmutateNumber, cmutateNumber,\
                scalemutatateNumber = 0, 0, 0, 0, 0
            while (numberMutations > 0):
                will = random.uniform(0,1)
                if will < gmutateprob:
                    gmutateNumber += 1
                elif will < gmutateprob+INSmutateprob:
                    INSmutateNumber += 1
                elif will < gmutateprob+INSmutateprob + DELmutateprob:
                    DELmutateNumber += 1
                elif will < gmutateprob+INSmutateprob + DELmutateprob + cmutateprob: 
                    cmutateNumber += 1
                else:
                    scalemutatateNumber += 1
                numberMutations -= 1
            if len(self.children_pop) == 0 :
                print "Number mutations (", numberMutations, "): ", gmutateNumber, ", ",\
                    INSmutateNumber, ", ", DELmutateNumber, ", ", cmutateNumber, ", ", scalemutatateNumber
            
            if i == len(self.population) or second:
                i = random.randrange(len(self.population))
                second = True
            #i = random.randrange(len(self.population))
            #while self.population[i].fitness/self.avg_fitness 
            #aChild = self.population[i].copy()
            #print "fit: ", aChild.fitness
            #while aChild.selection_measure < random.randint(-1,self.tournament_size):
            #while aChild.fitness < self.avg_fitness + random.normalvariate(0,20):
            curr_popularity = max(self.population, key=attrgetter('popularity'))
            while (i < random.randint(5,len(self.population)) or i in used or self.population[i].popularity > curr_popularity) and second:
                i = random.randrange(len(self.population))
                curr_popularity = self.population[i].popularity
            if second:                
                used.append(i)
                
            self.population[i].popularity += 1
            aChild = self.population[i].copy()
            
            
            i += 1

            #print "finished loop..."
            # Pick random parent, copy, mutate in three ways, add to pop
            #self.children_pop.extend(self._gaussianMutate(numGaussians))
            self._gaussianMutate(aChild, gmutateNumber)
            self._insertDeleteMutate(aChild, INSmutateNumber, DELmutateNumber)
            self._crossover(aChild, cmutateNumber)
            self._scaleMutate(aChild, scalemutatateNumber)
            
            #self.children_pop.extend(self._insertDeleteMutate(numInsertDel))
            #self.children_pop.extend(self._crossover(numCrossover))
            self.children_pop.append(aChild)
            
        print "Finished mutate"
        #self.children_pop.extend(self._scaleMutate())
        # mutate

    def nextgen(self, data):
        print "            data size: ", data.shape, ", children: ", len(self.children_pop)
        # combine parents and children
        #self.population = [max(self.population, key=attrgetter('fitness'))]#self.population[0:3]
        self.best_hid_bias = .9*self.best_hid_bias + .1*self.population[0].hid_bias
        self.best_vis_bias = .9*self.best_vis_bias + .1*self.population[0].vis_bias
        
        print "   ... Calculating fitnesses.." # SLOW part
        
        sumfit = 0
        oneFifthRule = 0
        childrenToKill = []
        for i in xrange(len(self.children_pop)-1, -1, -1):
            currFit = self.children_pop[i].calculateFitness(data)
            if i < len(self.population) and currFit > self.population[i].fitness:
                self.population.pop(i)
            elif i < self.pop_size:
                childrenToKill.append(i)
                
            if currFit > self.best_fitness:
                oneFifthRule += 1                
            sumfit = sumfit + currFit
        #childrenToKill.reverse()
        #print "len ctk: ", len(childrenToKill)
        for i in childrenToKill:
            self.children_pop.pop(i)
        print "Num pop: ", len(self.population), ", num child: ", len(self.children_pop)
        self.population.extend(self.children_pop)
        
        self.avg_fitness =  .8 * sumfit / len(self.children_pop) + .2*self.avg_fitness
        oneFifthRule = (oneFifthRule * 1.0) / float(len(self.children_pop))
        if oneFifthRule < .15 and self.mutationProbability > .4:
            self.mutationProbability -= (.001*self.mutationProbability)
        elif oneFifthRule > .3 and self.mutationProbability < 1:
            self.mutationProbability += .01
        else:
            self.mutationProbability += (.01*self.mutationProbability)
        print "One fifth prob: ", oneFifthRule, ", ", self.mutationProbability

        print "   ... Tournament.."
        random.shuffle(self.population)

        for t in xrange(self.tournament_size):
            for i in xrange(len(self.population)):
                if t == 0:
                    self.population[i].selection_measure = 0
#                    self.population[i].selection_measure = (self.population[i].fitness-self.avg_fitness)\
#                                /abs(self.avg_fitness) > random.normalvariate(0,1)
#                else:
                fighter = random.randrange(len(self.population))
                self.population[i].selection_measure += int((self.population[i].fitness-self.population[fighter].fitness)\
                            /abs(self.avg_fitness) > random.normalvariate(0,1))
                                #random.normalvariate(0,1))
        
        best_fit_individual = max(self.population, key=attrgetter('fitness'))
        #best_fit_individual.selection_measure += 2
        self.best_fitness = best_fit_individual.fitness
        self.worst_fitness = min(self.population, key=attrgetter('fitness')).fitness
        
        self.population.sort(key=lambda x: x.selection_measure, reverse=True)
        self.population.insert(0, best_fit_individual)
        #self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.population = self.population[0:self.pop_size]
        self.pop_size = len(self.population)
        
        #return self.best_fitness
    
    def learn(self, visible, children_multiplier = None):#, learning_rate=0.2, cdk=1):
        #self.children_multiplier = 3
        print "Learning.."
        if children_multiplier is not None:
            self.children_multiplier = children_multiplier
        print "Mutating..."
        print "    num children: ", str(self.children_multiplier*self.pop_size)
        self.mutate(math.ceil(self.children_multiplier*self.pop_size), .35, .3, .28, .07)#self.population)
        #self.mutate(self.children_multiplier*self.pop_size, .5,.3,.3,.3) #70,10,20)#self.population) #25/150 not mutated..
        

        print "Producing next gen..."
        self.nextgen(visible)
        print "..."
        return self.best_fitness
        
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
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_avfit_" + str(abs(int(self.avg_fitness)))+ "_pop.png")
            else:
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_avfit_" + str(abs(int(self.avg_fitness)))+ "_pop_"+saveType+".png")
            #str(self.data_loc) + 
        print "Filename: ", "epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_avfit_" + str(abs(int(self.avg_fitness)))+ "_pop.png"

        plt.cla()
        plt.hold(False)
        #plt.close(fig)
        #plt.close("all")
        #gc.collect()
        
            
    def visualizeRecons(self, data, todo = 'show', saveType = None):
        
        popSize = len(self.population)
        numRecons = len(data)
        fig = plt.figure(1, (80., 80.))
        fig.clf()
        grid = ImageGrid(fig, 111, # similar to subplot(111)
                        nrows_ncols = (numRecons, popSize), # creates grid of axes
                        axes_pad=0.1, # pad between axes in inch.
                        )
        #print "Size of fig: ", sys.getsizeof(fig)
        for i in range(numRecons):
            for j in range(popSize):
                grid[i*popSize+j].imshow(self.population[j].reconstruct(data[i], 2)) # The AxesGrid object work as a list of axes.
        #fig.savefig("Hello.png")
        if todo == 'show':
            print "Showing..."
            plt.show()
        elif todo == 'save':
            if saveType is None:
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_avfit_" + str(abs(int(self.avg_fitness)))+ "_recons.png")
            else:
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_avfit_" + str(abs(int(self.avg_fitness)))+ "_recons"+saveType+".png")
                
        print "Filename: ", "epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_avfit_" + str(abs(int(self.avg_fitness)))+ "_recons.png"

        plt.cla()
        plt.hold(False)
        #plt.close(fig)
        #plt.close("all")
        #gc.collect()
        
def testEvolution():
    ''' Second thread used to break out of training loop'''
    #thr = threading.Thread(target=running_func)
    #thr.start()
    global running
    
    #os.chdir("C:\\Users\\Ryan\\Documents\\Python\\EvolveRBM")
    
    ''' Get training data'''
    #training_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\male\\Zoom\\F", "C:\\Users\\Ryan\\Documents\\\SWAG_TRAINING\\female\\Zoom\\F"])
    mn = MNIST()    
    training_data, classes = mn.load_training()

    #training_data = np.asarray(training_data) #[0:50])
    #training_data = np.reshape(training_data, [len(training_data),28,28])
    #training_data = (training_data * 1.0)/255 #training_data.max()

    # RIZKI
    training_data = np.asarray(training_data, float)
    training_data = training_data / 255.0
    training_data = np.reshape(training_data, (len(training_data), 28, 28))      


    #cv2.imshow('image', training_data[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #imsave("Images\\reconL0_"+str(20)+".jpg", training_data[0])
    saveType = "genone_lpop" #20
    #np.random.shuffle(training_data)
    
    #training_data = training_data[0:50000]
    np.random.seed(101)
    np.random.shuffle(training_data)
    training_data = training_data[0:10000]#*1.0)/255
    # TRAIN SEED 101, TEST SEED (from remainder) 103
    print 'Test ConvRBM'
    rLayers = 10
    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
#    r = ConvolutionalEvolution(rLayers, [15,15], [2,2], False) #ConvolutionalEvolution(2, [3, 3], [2, 2])
    
    '''Params:  initPop = 100,  children_mult = 4, num_filters=20,      tourney=5, 
                filter_size=12, num_gaussians=5,   mutation_prob = .7
    '''

    #t = EvolveTrainer(50, 1.02, rLayers, 1, num_gaussians=5, mutation_prob = .7)

    # RIZKI
    t = EvolveTrainer(initPop = 50, children_mult = 1.02, num_filters=10, tourney=1, filter_size=12, num_gaussians=1, mutation_prob = .7)
    
#    t = pickle.load(open("trainer"+"_"+saveType+".p", "rb"))#
#    for i in range(len(t.population)):
#        r = pickle.load(open("conv"+str(i)+"_"+saveType+".p", "rb"))
#        r.filter_changed = np.ones(r.num_filters, dtype=bool)
#        t.population[i] = r
    #t.children_multiplier = 1.1
#    t.mutationProbability =.7
    #t.tournament_size = 3
    #t.pop_size = 10
    #t.children_pop
    #rLayers = r.num_filters
    print "Visualizing..."
    #t.visualizePop(todo='save', saveType=saveType)
    #t.visualizeRecons(training_data[0:10])
    #return 0
    #for i in range(len(t.population)):
    #    t.population[i].fitness = -9999
    r = t.population[0]
    batchsize = 30
    #t.stats = []
    #t.best_fitness = 100
    #t.best_hid_bias = np.zeros(r.num_filters)
    #t.best_vis_bias = np.zeros(1)
    '''Trainer(rbm, momentum=0., l2=0., target_sparsity=None):'''
    #t = ConvolutionalTrainer(r,.5, 0, .005) #changed from .005 to .05
    print 'Training...'
    for i in range(rLayers):
        imsave(os.path.join("Images", "weights_init_" +saveType+"_"+str(i)+".jpg"), r.weights[i])
    ''' Training for first layer'''
    #startOff = 0
    #if len(t.stats) > 0 :
    #    startOff = t.stats[len(t.stats)-1][1]
    bfit, prev_bfit = 0, 0
    j = 0
    #t.mutationProbability = .8
    start = time.clock()
    print "t.epoch: ", str(t.epoch)
    for i in range(t.epoch, 5000):
        t.epoch = i
        batchsize = 30 + int(i//2)
        ''' Get NEW training data'''
        global avgRE, minRE1
        #np.random.shuffle(training_data)
        #for j in range(t.data_loc, training_data.shape[0], batchsize):
        t.data_loc = j
        print "Epoch: ", str(i), " Data: ", str(j), "/", str(training_data.shape[0]),\
                " batchsize: ", str(batchsize)
        #print "bat size: ", training_data[j:j+batchsize].shape
        prev_bfit = bfit
        
        
        bfit = t.learn(training_data[j:j+batchsize]) #, 20)
        
        elapsed = (time.clock() - start)
        t.stats.append((i, j, t.avg_fitness, t.best_fitness, t.worst_fitness, elapsed))
        print "stats size: ", sys.getsizeof(t.stats)
        #avgRE = r.get_avg_error(training_data[j])
        print "num children: ", len(t.children_pop)
        #print "pop size: ", t.pop_size
        r = t.population[0]
        
        print 'Avg Fitness: ', t.avg_fitness, ', Best Fit: ', \
                str(t.best_fitness), ', Worst Fit: ', str(t.worst_fitness)
        print "id: ", str(r.id), " averages: (w, h, v) ", str(r.weights.mean()), ", ", str(r.hid_bias.mean()), ", ", str(r.vis_bias)
        
        if j % 1 == 0 :
            ''' Reconstruct image for one layer'''   
            k = random.randint(0, batchsize)                 
            minRecon = r.reconstruct(training_data[k], 2)
            rerror = r.get_avg_error(training_data[k])
            if bfit > prev_bfit and (i+j) % 1 == 0:
                #rerror < t.min_reconerror and i+j % 1 == 0:
                print "Visualizing..."
                t.min_reconerror = rerror
                #if j % 100:
                t.visualizePop(todo='save', saveType=saveType)
                t.visualizeRecons(training_data[j:j+10], todo='save', saveType=saveType) #data
            #elif j % 1 == 0 and batchsize < 100:
                #batchsize += 1
            #minRecon = minRecon / minRecon.max() * 255
            with lock:
                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_0.jpg"), training_data[k]*255)
                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_1.jpg"), minRecon*255)
                #t.visualizePop('save')
        # Save parameters occassionally 
        if j % 5*batchsize == 0 :
            for k in range(len(t.population)):
                r = t.population[k]
                pickle.dump(r, open("conv"+str(k)+"_"+saveType+".p", "wb"))
            print 'Saving layer 1 weights'
            pickle.dump(t, open("trainer"+"_"+saveType+".p", "wb"))
            
            for k in range(rLayers):
                imsave(os.path.join("Images", "weights_iter"+str(i)+saveType+str(k)+".jpg"), r.weights[k])
        if not running:
            with lock:
                print 'First break'
                print 'Breaking on running (in)'
            break
        # END SECOND INDENT
            #t.data_loc = 0
        #if abs(oldRE - avgRE) < .0001:
        #    break
            #if not running:
            #    print 'Second break'
            #    print 'Breaking on running (out)'
            #    break
                #print 'shape: ', r.hidden_sample(training_data[j]).shape
    #with lock:
    #    print 'Joining threads...'
    #thr_train.join()
    t.visualizePop(todo='save', saveType=saveType)
    t.visualizeRecons(training_data[j:j+10], todo='save', saveType=saveType) #data
    for i in range(len(t.population)):
        r = t.population[i]
        pickle.dump(r, open("conv"+str(i)+"_"+saveType+".p", "wb"))
    print 'Saving layer 1 weights'
    pickle.dump(t, open("trainer"+"_"+saveType+".p", "wb"))
    #t.visualizePop()
    #t.visualizeRecons(training_data[0:batchsize])
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
    #cProfile.run('testEvolution()')
    print "fini"
    
   