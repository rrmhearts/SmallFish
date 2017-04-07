# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:16:26 2013

@author: Ryan based on Leif's wrong implementation
"""
from RBM import RBM
from utils import *
from PIL import Image
from scipy.misc import imsave
from scipy import stats
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.filter.rank import entropy
from skimage.morphology import disk

from scipy.cluster.vq import *
from MyError import NaNError, MyError
import numpy as np, itertools
import logging, threading, pickle
import sys, os, math, time, csv
import weka, random, bisect
import copy, operator
import cProfile
import numpy.random as rng
from datastorage import Data #dl_data
from mnist import MNIST
from operator import attrgetter

import matplotlib as mpl
#print mpl.get_backend()
#mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
corr = scipy.signal.correlate 

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
    best_entropy = 0
    def __init__(self, num_filters, filter_shape, pool_shape, binary=True, num_gaussians=5, best_guy=None):
        '''
            num_filters: Number of filters for the CRBM
            filter_shape: weight shape [x,y]. usually x=y
            pool_shape: pooling region. 
            binary: whether the input data is binary or gaussian. Effects visible expectation
            num_gaussians: number of gaussians to init in each filters
        '''
        self.bernoul = .3
        self.id = ConvolutionalEvolution.count
        ConvolutionalEvolution.count += 1
        self.fitness = None
        self.error = None
        self.selection_measure = 0
        self.popularity = 0
        self.filter_changed = np.ones(num_filters, dtype=bool)
        self.num_filters = num_filters
        self.weights_size = np.append(num_filters, filter_shape)
        self.weights = np.random.rand(*self.weights_size)
        self.filtercorrelation = 1
        self.error = -1
        self.sparsity = np.zeros(self.num_filters)
        
        ''' Avg Vis ON / num_filters /2. Each area of VIS should be activiated by only one num_filter '''
        self.target_sparsity = .1 / num_filters  #* ( 2./filter_shape[0] )#.1 * 28 / filter_shape[0]) #3 #.05
        self.entropy = 26
        self.amplitude = random.uniform(0,1)#1.0 / num_filters * random.uniform(0,1)
        self.overhidden = 2
        self.weights_binary = np.array([gaussian_filter(np.random.rand(*filter_shape),1)+.05 < \
                                gaussian_filter(self.weights[i], 1) for i in range(self.num_filters)])
        self.update_weights()
#        print self.weights
        ''' Visible bias'''
        self.vis_bias = rng.randn(1)   # Scalar visual bias
        # np.zeros(1) #
        ''' Bias for each hidden filter'''
#        self.hid_bias = rng.randn(num_filters)   # Scalar bias for each filter
        self.hid_bias = np.zeros(num_filters)
        
        if best_guy is not None:
            self.hid_bias = best_guy.hid_bias * random.normalvariate(1,.5)
            self.vis_bias = best_guy.vis_bias * random.normalvariate(1,.5)
#            self.weights_binary = gaussian_filter(self.weights_binary,2)-self.weights_binary.mean() > \
#                                    gaussian_filter(self.weights_binary,2) + gaussian_filter(np.random.rand(*self.weights_size),2) \
#                                    -2*(best_guy.weights_binary.mean())
            self.update_weights
            
        # -1 * np.ones(num_filters) #
        ''' Sigmoid or gaussian for input type'''
        self._visible = binary and sigmoid or identity # gaussian #identity
        #self._visible = binary and sigmoid or gauss
        self._pool_shape = pool_shape
    def update_weights(self):
        ''' Set weights according to genome. Map genome to weight matrices
            Only upddates fiters that have corresponding changes in the genome.
        '''
        self.weights = np.zeros(self.weights_size)
        for i in range(len(self.weights)):
            for x in range(len(self.weights[i])):
                for y in range(len(self.weights[i,x])):
                    if self.weights_binary[i,x,y]:
                        g = gaussianf(self.amplitude, x, y, .6, .6, 0)
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
        
    def hidden_expectation(self, visible, bias=0.):
        '''Equation 14 in Lee paper
           Given visible data, return the expected hidden unit values.
        '''
        activation = np.array([
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid') # flipped h & v
            for k in range(self.num_filters)]).T + self.hid_bias + bias
        act = activation.T
        
        act = np.exp(act)
        
#        if numpy.isnan(act):
#            print "NaN found in hidden_expectation: line 171"
        numpy.nan_to_num(act)
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
        
    def calculateEntropy(self):
        
        self.entropy = 0
        for i in range(len(self.weights)):
            norm_weights = (self.weights[i]-self.weights[i].min())/\
                           (self.weights[i].max()*1.0-self.weights[i].min())
            self.entropy += entropy(norm_weights, disk(12)).mean()
#            self.entropy += self.entropyF(self.weights[i].sum(axis=0))+self.entropyF(self.weights[i].sum(axis=-1))
        self.entropy /= len(self.weights)
        return self.entropy
        
    def calculateFitness(self, data):
        ''' Fitness function for machine based on the negative reconstruction error.
        
            fitness = 1 - error - overhidden - entropy
        '''
        self.update_weights()
        self.error = self.get_reconstruction_error(data)
        
        all_active = np.zeros((self.num_filters))
        # Prevent weights from dropping out
#        for i in range(len(self.weights)):
#            self.fitness -= 9999*(self.weights[i].mean(axis=-1).mean(axis=-1) <= self.amplitude/10)
        blobs_one = 0.
        for k in self.weights_binary:            
            bbs, nbbs = ndimage.label(k>0)
#            print nbbs
            if nbbs == 0:
                blobs_one += 3.
            elif nbbs == 1:
                blobs_one += 0
            else:
                blobs_one += abs(1.-nbbs)
        blobs_one /= len(self.weights)
        
        self.overhidden = 0
        self.sparsity *= .99
        num_uh_used = 0
        for visible in data:
            hidden = self.hidden_sample(visible)
#            hmean = hidden.mean()
            all_active += 1.0*(hidden.mean(axis=-1).mean(axis=-1))
            num_uh_used += np.mean(1.*(hidden.mean(axis=-1).mean(axis=-1)>0))
#            sabs_mean = abs(self.sparsity.mean())
            if self.sparsity.mean() == 0:
                self.sparsity += (self.target_sparsity - hidden.mean(axis=-1).mean(axis=-1))
            elif abs(self.target_sparsity - hidden.mean()) > self.target_sparsity:
                self.sparsity += .01*(self.target_sparsity - hidden.mean(axis=-1).mean(axis=-1)) / len(data)#axis=-1).mean(axis=-1) 
            else:
                self.sparsity += .01*(np.random.rand(self.num_filters) - self.sparsity)/100
            self.hid_bias += self.sparsity #+ np.random.rand(self.num_filters)/10000
            #abs(self.target_sparsity-hmean)/sabs_mean*self.sparsity
            
            temp_hidden = 0
            xor_hidden = 0
            for i in range(len(hidden)):
                for j in range(i+1, len(hidden)):
                    temp_hidden += np.logical_and(hidden[i], hidden[j]).mean() * 1.0 /(len(hidden)*(len(hidden)-1)) * 2 
                xor_hidden += 1.*hidden[i] if hidden[i].mean() <= self.num_filters*self.target_sparsity else 0.*hidden[i]
                    #- np.sum(np.logical_xor(hidden[i], hidden[j]))
            xor_term = min((xor_hidden==1).mean(), self.target_sparsity * len(hidden)) / (len(hidden)) #
            
            
#            if abs(2000*(temp_hidden - xor_term*num_h_used )) > 500:
#                inside = 500*(temp_hidden - xor_term*num_h_used)/abs(temp_hidden - xor_term*num_h_used)
#                print "Inside exp is too large! ", (temp_hidden - xor_term*num_h_used)
#            else:
#            print "temp hidden: ", temp_hidden
#            print "xor hidden: ", xor_term
#            print "num h used: ", num_h_used
            
            
            inside = 20000*(3*temp_hidden - 1*xor_term)#*num_uh_used)
            expn = np.exp(-inside)
            temp_hidden = 1 / (expn+1) / len(data) #expn / (expn+1) / len(data)
            
            
#            if np.isnan(temp_hidden):
#                print "temp hidden: ", temp_hidden
#                print "xor_term: ", xor_term
#                print "num_h_used: ", num_h_used
                
#                temp_hidden = 0
#            if hmean == 0: #< self.target_sparsity/len(hidden):
#                self.fitness -= .8*abs(self.fitness)
                
                
            self.overhidden += temp_hidden
        num_uh_used /= len(data)
        num_uh_used = abs(num_uh_used-.6) /(1.-.6)
        
        all_active /= len(data)
        all_active = abs(self.target_sparsity*.6 - all_active.min())
#        print "overhidden: ", self.overhidden
            #print "overhidden: ", self.overhidden
#        if abs(self.sparsity.mean() > 0:        
#            self.fitness -= 10*self.sparsity.mean()
        if np.isnan(self.error) or np.isnan(self.overhidden) or np.isnan(num_uh_used) or np.isnan(all_active):
            print "ERROR!!!! FITNESS FUNTION!!"
#        self.fitness -= 500*self.overhidden
#        print "num used: ", num_uh_used
#        print "nused: ", num_uh_used
#        print "active: ", all_active
#        print "blobs one: ", blobs_one
        self.fitness = 100.0  - 60*self.error#*num_uh_used
        self.fitness -= 20*self.overhidden 
        self.fitness -= 5000*(4*num_uh_used + all_active) #*self.error
        self.fitness -= blobs_one
#        self.fitness -= 1000*all_active
        self.fitness -= abs(self.sparsity).max()
#        self.fitness -= self.calculateEntropy()/10
        
        return self.fitness
       
class EvolveTrainer():
    
    def __init__(self, initPop = 100, children_mult = 4, num_filters=20, tourney=5, filter_size=10, num_gaussians=5, mutation_prob = .7):
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
        self.avg_fitness = -1
        self.best_fitness = -1
        self.worst_fitness = -1
        self.min_reconerror = 1000
        self.children_pop = []
        self.best_hid_bias = np.zeros(num_filters)
        self.best_vis_bias = np.zeros(1)

        self.epoch = 0
        self.data_loc = 0
        self.stats = []
        
        self.diversity_cutoff = 5.0
        self.percent_kept = .8
        self.deviation = 1
        self.filter_size = filter_size
        self.num_filters = num_filters
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
        ''' Gaussian mutate on existing gauss genes'''
        for j in range (0, number): #len(self.population)):
            # filter
            ''' Potentially updating the amplitude...'''
#            if np.random.randint(0,1)==0:                               
#                                
#                aChild.weights_binary = np.array([gaussian_filter(aChild.weights_binary[i]*1.0-aChild.weights_binary[i].mean(),1) > \
#                                gaussian_filter(np.random.rand(*aChild.weights_size[1:])-.5, 1)+.1 for i in range(aChild.num_filters)])
#            else: #elif np.random.randint(0,1)==0:
            aChild.weights_binary = np.array([np.logical_and(ndimage.binary_dilation(aChild.weights_binary[i]), \
                    ndimage.binary_dilation(self.population[np.random.randint(len(self.population)-1)].weights_binary[i])) \
                    if aChild.weights_binary[i].mean() < .4 else self.population[np.random.randint(len(self.population)-1)].weights_binary[i]\
                    for i in range(aChild.num_filters)])
#            else:
#                aChild.weights_binary = np.array([gaussian_filter(aChild.weights_binary[i]*1.0,1) > aChild.weights_binary[i].mean() + \
#                                                                  2*aChild.weights_binary[i].std() for i in range(aChild.num_filters)])
#                temp = gaussian_filter(aChild.weights_binary[i],1)*aChild.weights_binary[i]
#                aChild.weights_binary[aChild.weights_binary== temp.min()] = 0
                                        

        
    def _insertDeleteMutate(self, aChild, numberInserts, numberDeletes):
        ''' Insert and delete genes from genome '''

###            ''' CONSTRAINT: Emphasize border regions'''
        probMat = np.array([abs ( gaussian_filter(aChild.weights[i], 1) - gaussian_filter(aChild.weights[i], 3)) for i in range(aChild.num_filters)])
        probMat = (probMat-probMat.min()) / (probMat.max()-probMat.min())

        for i in range (numberInserts):

            uni = np.random.rand(*aChild.weights.shape)

            point = probMat*2 - uni*2 - 4.*aChild.weights_binary
            if aChild.weights_binary.mean() < .5:
                aChild.weights_binary[point == point.max()] = 1
            else:
                aChild.weights_binary[point == point.max()] = 0

        for i in range (numberDeletes):
#            old = np.array(aChild.weights_binary)
#            negMat += (aChild.weights_binary.mean()-negMat.mean())
            uni = np.random.rand(*aChild.weights.shape)

            point = probMat*2 - uni*2 + 4.*aChild.weights_binary
#            point_maxs = point.max(axis=-1).max(axis=-1)
            if aChild.weights_binary.mean() > .1:
                aChild.weights_binary[point == point.max()] = 0
            else:
                aChild.weights_binary[point == point.max()] = 1
                
    def _crossover(self, aChild, number):
        ''' Crossover on gene level (recently changed) '''
        # make sure using right term population THE RBM not pop THE WEIGHTS
        for i in range(number):
            #i = i % len(self.population)
            cross = random.randrange(len(self.population))
            c1 = self.population[cross].copy()
#            cross = random.randrange(len(self.population))
#            c2 = self.population[cross].copy()
            
            filt_start = random.randrange(len(self.population[cross].weights))#-1)
#            filt_end = random.randint(filt_start+1, 
#                                      len(self.population[cross].weights)-1)
#            aChild.weights_binary[filt_start:filt_end] = c1.weights_binary[filt_start:filt_end]
            aChild.weights_binary[filt_start] = c1.weights_binary[filt_start]

            
    def mutate(self, numberChildren, gmutateprob=None, INSmutateprob=None, DELmutateprob=None, cmutateprob=None, scalemutatateprob=None):#, pop):
        ''' Mutate: select number of mutations and which kind they are. Create and Mutate children until reach numberChildren'''
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
        second = True #False
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
                else: #elif will < gmutateprob+INSmutateprob + DELmutateprob + cmutateprob: 
                    cmutateNumber += 1
                numberMutations -= 1
            if len(self.children_pop) == 0 :
                print "Number mutations (", numberMutations, "): ", gmutateNumber, ", ",\
                    INSmutateNumber, ", ", DELmutateNumber, ", ", cmutateNumber, ", ", scalemutatateNumber
            
            if i == len(self.population) or second:
                i = random.randrange(len(self.population))
                second = True

            i = random.randrange(len(self.population))
#            while i > random.randrange(len(self.population))+1:
#                i = random.randrange(len(self.population))
            aChild = self.population[i].copy()
            
            self._insertDeleteMutate(aChild, INSmutateNumber, DELmutateNumber)
            self._crossover(aChild, cmutateNumber)
            self._gaussianMutate(aChild, gmutateNumber)

            #self._scaleMutate(aChild, scalemutatateNumber)
            if random.randint(0,10) == 1:
                aChild.vis_bias = .8*aChild.vis_bias + 2*rng.randn(1) + .2*self.best_vis_bias
            if random.randint(0,10) == 1:
                sc = random.normalvariate(.0,.01) 
                if aChild.amplitude < 1 or sc < 0:
                    aChild.amplitude += sc
                
                
            if random.randint(0,10) == 1 and aChild.bernoul < .75:
                aChild.bernoul += .05
            elif random.randint(0,10) == 1 and aChild.bernoul > .25:
                aChild.bernoul -= .05
            else:
                pass
#            print "HID BIAS? ", aChild.hid_bias
#            print "sparsity? ", aChild.sparsity
            #self.children_pop.extend(self._insertDeleteMutate(numInsertDel))
            #self.children_pop.extend(self._crossover(numCrossover))
            self.children_pop.append(aChild)
            
        print "Finished mutate"
        #self.children_pop.extend(self._scaleMutate())
        # mutate
    def removeDuplicates(self, data = None):
        ''' CONSTRAINT: Remove duplicate individuals from population. Always less fit versions. '''
        i, j = 0, 0
        
        removed_count = 0
        pop_size_before = len(self.population)
        # Pre-screen
        while i < len(self.population):
            if self.population[i].weights.mean(axis=-1).mean(axis=-1).min() <= 0.001 \
                or self.population[i].weights_binary.mean(axis=-1).mean(axis=-1).min() <= 0.001 \
                or stats.mode(self.population[i].weights.mean(axis=-1).mean(axis=-1))[0][0]<=0.001 \
                or stats.mode(self.population[i].weights_binary.mean(axis=-1).mean(axis=-1))[0][0]<=0.001:
                
                self.population.pop(i)
                removed_count += 1
            else:
#                print stats.mode(self.population[i].weights_binary.mean(axis=-1).mean(axis=-1))[0][0]
                i+=1
        i = 0
        if len(self.population) ==0:
            print "ERROR! POP DIED!!"
        while i < len(self.population)-1:
            norm_i = self.population[i].weights /np.linalg.norm(self.population[i].weights)

#            blobs_good = True
#            for k in self.population[i].weights:            
#                bbs, nbbs = ndimage.label(k>0)
#                blobs_good = False if nbbs == 0 or nbbs > 3 else blobs_good

            while len(self.population) > 5 and (np.isnan(norm_i).any() or self.population[i].weights.mean() <= self.population[i].amplitude/8 \
                                            or self.population[i].weights_binary.mean() > .6 or self.population[i].weights_binary.mean() < .1):
#                                            or not blobs_good):
#                    print "    In delete..."
                    self.population.pop(i)
                    
                    
                    if i < len(self.population):
                        norm_i = self.population[i].weights /np.linalg.norm(self.population[i].weights)
                    else: 
                        break
                    removed_count += 1
#                    print "  Deleted? ", self.population[i].weights.mean()2
            #else: # Compare everything to i
            j = i + 1
            howmany = 2 #(1.-self.diversity_cutoff)/self.deviation
            while j < len(self.population):
                norm_j = self.population[j].weights /np.linalg.norm(self.population[j].weights)
                curr_diversity = float(corr(norm_i, norm_j, 'valid'))
#                print "curr div: ", curr_diversity
                if curr_diversity > self.diversity_cutoff+abs(howmany)*self.deviation \
                    or np.isnan(norm_j).any() or self.population[j].weights.mean() <= self.population[j].amplitude/8\
                    or self.population[j].weights_binary.mean(axis=-1).mean(axis=-1).max() > .6 \
                    or self.population[j].weights_binary.mean(axis=-1).mean(axis=-1).min() < .1:
                    self.population.pop(j)
                    removed_count += 1
                else:
                    j += 1
                    self.diversity_cutoff = .999* self.diversity_cutoff  #+ .001 * curr_diversity
                    self.deviation = .999*self.deviation
    #                if self.percent_kept < .8:
                    if not math.isnan(curr_diversity):
                        self.diversity_cutoff += .001 * curr_diversity
                        new_deviant = self.deviation**2 + (self.diversity_cutoff-curr_diversity)**2
                        new_deviant = math.sqrt(new_deviant/2)
                        self.deviation += .001*new_deviant
                howmany = ((1.-self.diversity_cutoff)/self.deviation/len(self.population) + howmany*(len(self.population)-1)/len(self.population))

            i += 1
        # END LOOPING
            
        self.percent_kept =(pop_size_before - removed_count)*1.0 / pop_size_before

        if self.percent_kept < .10 or len(self.population) < self.pop_size:

            ''' CONSTRAINT: Add new members if population falls below pop_size'''
            for i in range(0, int(self.pop_size*(1.0-self.percent_kept)+1)):
                self.population.append(ConvolutionalEvolution(self.num_filters, [self.filter_size, self.filter_size], [2, 2], num_gaussians=15, best_guy=self.population[0]))
                if data is not None:
                    self.population[-1].calculateFitness(data)
#        elif self.diversity_cutoff < 2 or self.deviation < 0.1:
            #self.diversity_cutoff += .1
#            self.deviation *= 1.5

        if removed_count > 0:
            #print " Ex. corr: ", float(corr(self.population[0].weights, self.population[1].weights, 'valid'))
            print "..  Removed some"
            print "    Kept %: ", self.percent_kept
            print "    Div cut: ", self.diversity_cutoff
            print "    St Dev: ", self.deviation
            print "    Curr div: ", curr_diversity
        else:
            print "    None removed"
            print "    Div cut: ", self.diversity_cutoff
            print "    St Dev: ", self.deviation
            print "    Curr div: ", curr_diversity
            
    def nextgen(self, data):
        ''' Produce next generation '''
        print "            data size: ", data.shape, ", children: ", len(self.children_pop)
        # combine parents and children
        #self.population = [max(self.population, key=attrgetter('fitness'))]#self.population[0:3]
        self.best_hid_bias = .9*self.best_hid_bias + .1*self.population[0].hid_bias
        self.best_vis_bias = .9*self.best_vis_bias + .1*self.population[0].vis_bias
        
        print "   ... Calculating fitnesses.." # SLOW part
        
#        print "# children before: ", len(self.children_pop)
#        self.population.extend(self.children_pop)
#        self.removeDuplicates(data)
#        self.children_pop = list( set(self.population).intersection(self.children_pop) )
#        self.population = list(set(self.population) - set(self.children_pop))
#        print "# children after: ", len(self.children_pop)
        
        sumfit = 0
        oneFifthRule = 0
        for i in xrange(len(self.children_pop)-1, -1, -1):
            if self.children_pop[i].weights_binary.mean() > .05 and self.children_pop[i].weights_binary.mean() < .7:
                currFit = self.children_pop[i].calculateFitness(data)
                
                ''' CONSTRAINT: One-fifth rule (approximately)'''
                if currFit > self.best_fitness or currFit > self.population[1].fitness:
                    oneFifthRule += 1
                sumfit = sumfit + currFit
            else:
#                print "Killed zero guy."
                self.children_pop.pop(i)
        num_children = len(self.children_pop)

#        if oneFifthRule > 0: #Because of using pop[1]
#            oneFifthRule -= 1
        self.population.extend(self.children_pop)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        ConvolutionalEvolution.best_entropy = self.population[0].entropy
        ''' Here, remove repeats before setting new population'''
        print "# children before: ", len(self.children_pop)
        self.removeDuplicates(data)
        self.children_pop = list( set(self.population).intersection(self.children_pop) )
        print "# children after: ", len(self.children_pop)

        print "Best entropy? ", ConvolutionalEvolution.best_entropy
        ''' Set new population'''
        self.population = self.population[0:self.pop_size]

        print "Num pop: ", len(self.population), ", num child: ", len(self.children_pop)
        
        if num_children == 0:
            self.avg_fitness =  self.avg_fitness
        else:
            self.avg_fitness =  .5 * sumfit / num_children + .5*self.avg_fitness

        ''' CONSTRAINT: One-fifth rule (approximately)'''
        oneFifthRule = (oneFifthRule * 1.0) / float(len(self.population))
        if oneFifthRule < .05 : #and self.mutationProbability > .5:
            self.mutationProbability -= (.01*self.mutationProbability)
        elif oneFifthRule > .2 and self.mutationProbability < 1:
            self.mutationProbability += .001
        else:
            self.mutationProbability += (.01*self.mutationProbability)
        print "One fifth prob: ", oneFifthRule, ", ", self.mutationProbability
        print "   ... Tournament.."
#        random.shuffle(self.population)

#        for t in xrange(self.tournament_size):
#            for i in xrange(len(self.population)):
#                if t == 0:
#                    self.population[i].selection_measure = 0
##                    self.population[i].selection_measure = (self.population[i].fitness-self.avg_fitness)\
##                                /abs(self.avg_fitness) > random.normalvariate(0,1)
##                else:
#                fighter = random.randrange(len(self.population))
#                self.population[i].selection_measure += int((self.population[i].fitness-self.population[fighter].fitness)\
#                            /abs(self.avg_fitness) > random.normalvariate(0,1))
#                                random.normalvariate(0,1))
#        self.population.sort(key=lambda x: x.fitness, reverse=True)

        best_fit_individual = max(self.population, key=attrgetter('fitness'))
#        best_fit_individual.selection_measure += 2
#        self.population.sort(key=lambda x: x.selection_measure, reverse=True)
#        self.population = self.population[0:self.pop_size]
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        self.best_fitness = best_fit_individual.fitness
        self.worst_fitness = min(self.population, key=attrgetter('fitness')).fitness
    
    def learn(self, visible, children_multiplier = None):#, learning_rate=0.2, cdk=1):
        #self.children_multiplier = 3
        print "Learning.."
        if children_multiplier is not None:
            self.children_multiplier = children_multiplier
        print "Mutating..."
        print "    num children: ", str(self.children_multiplier*self.pop_size)
        self.mutate(math.ceil(self.children_multiplier*self.pop_size), .05, .3, .30, .35)# .3)#self.population)
        # .4, .25, .05, .3
        #self.mutate(self.children_multiplier*self.pop_size, .5,.3,.3,.3) #70,10,20)#self.population) #25/150 not mutated..
        

        print "Producing next gen..."
        self.nextgen(visible)
        print "..."
        return self.best_fitness
        
    def visualizePop(self, todo = 'show', popSelect = None, filterSelect = None, saveType = None):
        #plt.hold(False)
        
        if popSelect is None:
            popSize = min(len(self.population), 50) #min (len(self.population), 30)
        else:
            popSize = len(popSelect)
        if filterSelect is None:
            numfilters = len(self.population[0].weights)
        else:
            numfilters = len(filterSelect)
        fig = plt.figure(1, (80., 80.))
        fig.clf()
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
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_bestfit_" + str(abs(int(self.best_fitness)))+ "_pop.png")
            else:
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_bestfit_" + str(abs(int(self.best_fitness)))+ "_pop_"+saveType+".png")
            #str(self.data_loc) + 
        print "Filename: ", "epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_bestfit_" + str(abs(int(self.best_fitness)))+ "_pop.png"

        plt.cla()
        plt.hold(False)
        #plt.close(fig)
        #plt.close("all")
        #gc.collect()
        
            
    def visualizeRecons(self, data, todo = 'show', saveType = None):
        
        popSize = min(len(self.population), 50)
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
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_bestfit_" + str(abs(round(self.best_fitness,4)))+ "_recons.png")
            else:
                fig.savefig("epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_bestfit_" + str(abs(round(self.best_fitness,4)))+ "_recons"+saveType+".png")
                
        print "Filename: ", "epoch_"+str(self.epoch) + "_" +str(self.data_loc)+ "_bestfit_" + str(abs(round(self.best_fitness,4)))+ "_recons.png"

        plt.cla()
        plt.hold(False)
        #plt.close(fig)
        #plt.close("all")
        #gc.collect()kayleigh.duncan
        
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
    training_data = np.asarray(training_data)#[0:50])
    training_data = np.reshape(training_data, [len(training_data),28,28])
    training_data = (training_data * 1.0)/255 #training_data.max()
    #imsave("Images\\reconL0_"+str(20)+".jpg", training_data[0])
    saveType = "simple_genome" #20
    #np.random.shuffle(training_data)
    
    classes = np.asarray(classes)
    #training_data = training_data[0:50000]
    np.random.seed(101)
    np.random.shuffle(training_data)
    
    training_data = training_data[0:10000]#*1.0)/255
    classes = classes[0:10000]
    # TRAIN SEED 101, TEST SEED (from remainder) 103
    
#    training_data = training_data[classes==7]
#    print training_data.shape    
#    print classes.shape
    print 'Test ConvRBM'
    rLayers = 20
    '''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
#    r = ConvolutionalEvolution(rLayers, [15,15], [2,2], False) #ConvolutionalEvolution(2, [3, 3], [2, 2])
    
    #initPop = 100, children_mult = 4, num_filters=20, tourney=5, filter_size=12, num_gaussians=5,mutation_prob = .7):
    t = EvolveTrainer(50, 6, rLayers, 3, filter_size = 12, num_gaussians=10, mutation_prob = .4)
#    print "Working dir: ", os.getcwd()
#    t = pickle.load(open("trainer"+"_"+saveType+".p", "rb"))#
#    for i in range(len(t.population)):
#        r = pickle.load(open("conv"+str(i)+"_"+saveType+".p", "rb"))
##        r.filter_changed = np.ones(r.num_filters, dtype=bool)
#        t.population[i] = r
#    t.children_multiplier = 5
#    t.mutationProbability =.4
#    #t.tournament_size = 3
#    t.pop_size = 100
#    t.population = t.population[0:20]
    #t.children_pop
    #rLayers = r.num_filters
#    t.diversity_cutoff -= t.diversity_cutoff*.3
#    t.deviation -= .4*t.deviation
#    t.percent_kept = .6
    print "Visualizing..."
#    t.visualizePop(todo='save', saveType=saveType)
    #t.visualizeRecons(training_data[0:10])
    #return 0
#    for i in range(len(t.population)):
####        t.population[i].genelocations = np.zeros(t.population[i].weights_size, dtype=bool)
####        #t.population[i].entropy = 26
##        t.best_fitness = 0
##        t.population[i].bernoul = .3
##        if (t.population[i].weights.mean(axis=-1).mean(axis=-1)==0.0).any():
##            t.population[i].fitness -= 10
##        #t.population[i].vis_bias = 1
##        t.population[i].hid_bias+= .02 #[t.population[i].hidden==0] += .05
##    t.diversity_cutoff =.331 #-= 1.0001 *(t.diversity_cutoff)
##    t.deviation = 0#-10000.0 #-= 1.00001 * (t.deviation)
#        t.population[i].fitness -= 5 #= -9999
##    t.avg_fitness = 30
#    t.best_fitness -= 10
###        t.population[i].entropy = 25
    r = t.population[0]
    batchsize = 15
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
    j = random.randrange(0, batchsize)  
    #t.mutationProbability = .8
    start = time.clock()
    print "t.epoch: ", str(t.epoch)
    for i in range(t.epoch, 5000):
        t.epoch = i
        #batchsize = 2 #5 + int(i//6)
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
        #print "stats size: ", sys.getsizeof(t.stats)
        #avgRE = r.get_avg_error(training_data[j])
        print "num children: ", len(t.children_pop)
        #print "pop size: ", t.pop_size
        r = t.population[0]
        
        
        elapsed = (time.clock() - start)
        t.stats.append((i, j, t.avg_fitness, t.best_fitness, t.worst_fitness, elapsed, r.error, r.sparsity.mean(), r.overhidden))
        print 'Bernoull cut: ', r.bernoul
        print 'Avg Fitness: ', str(t.avg_fitness), ', Best Fit: ', \
                str(t.best_fitness), ', Worst Fit: ', str(t.worst_fitness)
        print "Error: ", r.error, ", entropy: ", r.entropy
        print "Amplitude: ", r.amplitude
        print "id: ", str(r.id), " averages: (w, h, v) ", str(r.weights.mean()), ", ", str(r.hid_bias), ", ", str(r.vis_bias)
        print "Target sparsity: ", r.target_sparsity
        print "Sparsity: ", r.sparsity
        print "Hidden overlap: ", r.overhidden
        print "Avg Hidden: ", r.hidden_expectation(training_data[j]).mean(axis=-1).mean(axis=-1)
        print "Max: ", np.max(r.weights), ", min: ", np.min(r.weights)
        #print "filtercorrelation: ", str(r.filtercorrelation)
        if j % 1 == 0 :
            ''' Reconstruct image for one layer'''   
            k = random.randrange(0, batchsize)                 
            minRecon = r.reconstruct(training_data[k], 2)
            rerror = r.get_avg_error(training_data[k])
            if bfit >= prev_bfit and (i+j) % 2 == 0:
                #rerror < t.min_reconerror and i+j % 1 == 0:

                print "Writing..."
                if len(t.stats ) > 0:
                    with open('convevolv_stats.csv', 'ab') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        spamwriter.writerows(t.stats)
                        t.stats = []
            if bfit >= prev_bfit and (i+j) % 3 == 0:
                print "Visualizing..."
                t.min_reconerror = rerror
                #if j % 100:
                t.visualizePop(todo='save', saveType=saveType)
                t.visualizeRecons(training_data[j:j+min(10,batchsize)], todo='save', saveType=saveType) #data
                
                
            hid = r.hidden_sample(training_data[k])
            exp = r.hidden_expectation(training_data[k])
            for y in range(len(r.weights)):
                imsave("expect"+str(y)+".jpg", exp[y])
                imsave("hidden"+str(y)+".jpg", hid[y])
            imsave("image"+str(k)+".jpg", training_data[k])
            imsave("recon"+str(k)+".jpg", minRecon)
            #elif j % 1 == 0 and batchsize < 100:
                #batchsize += 1
            #minRecon = minRecon / minRecon.max() * 255
#            with lock:
#                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_0.jpg"), training_data[k]*255)
#                imsave(os.path.join("Images", "reconL"+saveType+"_"+str(i*100+j)+"_1.jpg"), minRecon*255)
                #t.visualizePop('save')
        # Save parameters occassionally 
        if j % 1*batchsize == 0 :
            for k in range(len(t.population)):
                r = t.population[k]
                pickle.dump(r, open("conv"+str(k)+"_"+saveType+".p", "wb"))
            print 'Saving layer 1 weights'
            pickle.dump(t, open("trainer"+"_"+saveType+".p", "wb"))
        if j % 5*batchsize == 0 :
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
    
    print "Working dir: ", os.getcwd()
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
    
   