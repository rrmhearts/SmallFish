# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:40:00 2013

@author: Ryan based on Leif's RBM
"""
import logging
import numpy as np
import numpy.random as rng
import math
from utils import *
from MyError import NaNError, MyError

class RBM(object):
    '''A restricted boltzmann machine is a type of neural network auto-encoder.

    RBMs have two layers of neurons (here called "units"), a "visible" layer
    that receives data from the world, and a "hidden" layer that receives data
    from the visible layer. The visible and hidden units form a fully connected
    bipartite graph. To encode a signal,

    1. the signal is presented to the visible units, and
    2. the states of the hidden units are sampled from the conditional
       distribution given the visible data.

    To check the encoding,

    3. the states of the visible units are sampled from the conditional
       distribution given the states of the hidden units, and
    4. the sampled visible units can be compared directly with the original
       visible data.

    Training takes place by presenting a number of data points to the network,
    encoding the data, reconstructing it from the hidden states, and encoding
    the reconstruction in the hidden units again. Then, using contrastive
    divergence (Hinton 2002), the gradient is approximated using the
    correlations between visible and hidden units in the first encoding and the
    same correlations in the second encoding.
    '''

    def __init__(self, num_visible, num_hidden, binary=True, scale=0.001):
        ''' Initialize a restricted boltzmann machine.

            num_visible: The number of visible units.
            num_hidden: The number of hidden units.
            binary: True if the visible units are binary, False if the visible units
              are normally distributed.
        '''
        self.entropy = -1
        self.error = -1
        #if binary:
        self.weights = scale * rng.randn(num_hidden, num_visible)
        #else:
        #    self.weights = scale * np.abs(rng.randn(num_hidden, num_visible))
        self.hid_bias = 2 * scale * rng.randn(num_hidden)
        self.vis_bias = scale * rng.randn(num_visible)

        self._visible = binary and sigmoid or gaussian
        self.upper = None
        
    @property
    def num_hidden(self):
        return len(self.hid_bias)

    @property
    def num_visible(self):
        return len(self.vis_bias)
        
    def hidden_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected hidden unit values.'''
        ''' Weights(hidxvis) * vis '''
        #print 'hidExp Weights: ', self.weights
        return sigmoid(np.dot(self.weights, visible.T).T + self.hid_bias + bias)

    def visible_expectation(self, hidden, bias=0.):
        '''Given hidden states, return the expected visible unit values.'''
        ''' hid * Weights(hidxvis) '''
        return self._visible(np.dot(hidden, self.weights) + self.vis_bias + bias)

    def hidden_sample(self, visible, bias=0.):
        '''Sample hidden layer'''
        return bernoulli( self.hidden_expectation(visible))
        
    def visible_sample(self, hidden, bias=0.):
        '''Sample visible layer. '''
        if self._visible is sigmoid:
            return bernoulli( self.visible_expectation(bernoulli(hidden)))
        return self.visible_expectation(bernoulli(hidden))
        
    def iter_passes(self, visible):
        ''' Repeatedly pass the given visible layer up and then back down.

            Generates the resulting sequence of (visible, hidden) states. The first
            pair will be the (original visible, resulting hidden) states, followed
            by the subsequent (visible down-pass, hidden up-pass) pairs.
    
            visible: The initial state of the visible layer.
        '''
        while True:
            hidden = self.hidden_expectation(visible)
            hidden = bernoulli(hidden)
#            hidden = hidden > .4
            # Error checking
            if np.isnan(hidden).any():
                print 'NaN'
                hidden = np.nan_to_num(hidden)
                #print 'hidden: ', hidden
                #print 'hidden size: ', hidden.shape
                raise NaNError(hidden)
            yield visible, hidden

            visible = self.visible_expectation(bernoulli(hidden)) #hidden > .5)
#            visible = self.visible_expectation(hidden > .4)

    def reconstruct(self, visible, passes=1):
        '''Reconstruct a given visible layer through the hidden layer.

            visible: The initial state of the visible layer.
            passes: The number of up- and down-passes.
        '''
        for i, (visible, _) in enumerate(self.iter_passes(visible)):
            if i + 1 == passes:
                return visible
                
    def get_reconstruction_error(self, visible):
        ''' Get reconstruction error'''
        rerror = 0
        if len(visible.shape) == 3 and visible.shape[0] > 0:
            for i in range(visible.shape[0]):
                rvisible = self.reconstruct(visible[i],2)
                
                if np.isnan(rvisible).any():
                    print "NAN!!!!!!!!!!"
#                else:
#                    print "bottom.. ", (np.max(rvisible)- np.min(rvisible))
                rvisible = (rvisible - np.min(rvisible)) / (np.max(rvisible)- np.min(rvisible) + .00001)
                ''' EUclidean Distance'''
                

#                difference = rvisible - visible[i]
#                cerr = np.sum(difference*difference)
#                cerr /= (visible.shape[1]*visible.shape[2]) #normalize between 0-1
#                rerror = rerror + np.sqrt(cerr)
                ''' Percent area distance'''
                rvisible[rvisible > .5] = 1
                rvisible[rvisible <= .5] = 0
                
                visible[i][visible[i] > .5] = 1
                visible[i][visible[i] <= .5] = 0
                
                pos = np.logical_and(visible[i], rvisible)
                percent_pos = np.sum(pos)*1.0/np.sum(visible[i])
                
                neg = np.logical_and(1-visible[i], 1-rvisible)
                percent_neg = np.sum(neg)*1.0/np.sum(1-visible[i])
                rerror += (1 - percent_neg*percent_pos) + 2*np.sum(np.logical_and(1-visible[i], rvisible))*1.0/np.sum(1-rvisible)
            rerror = rerror / visible.shape[0]
        elif visible.shape[0] > 0:
            rvisible = self.reconstruct(visible,2)
            rvisible = (rvisible - np.min(rvisible)) / (np.max(rvisible)- np.min(rvisible))
#            difference = rvisible - visible
#            rerror = np.sum(difference*difference)
#            rerror /= (visible.shape[0]*visible.shape[1])
#            rerror = np.sqrt(rerror)
            rvisible[rvisible > .5] = 1
            rvisible[rvisible <= .5] = 0
            
            visible[visible > .5] = 1
            visible[visible <= .5] = 0
            
            pos = np.logical_and(visible, rvisible)
            percent_pos = np.sum(pos)*1.0/np.sum(visible)
            
            neg = np.logical_and(1-visible, 1-rvisible)
            percent_neg = np.sum(neg)*1.0/np.sum(1-visible)
            rerror += (1 - percent_neg*percent_pos) + np.sum(np.logical_and(1-visible, rvisible))*1.0/np.sum(1-rvisible)
            
        return rerror
        
    def get_avg_error(self, rerror=-1):
        ''' Get running average error'''
        
        # If given array, compute reconstruction error
        if type(rerror) is np.ndarray:
            rerror = self.get_reconstruction_error(rerror)
            
        # Update running average error
        if math.isnan(rerror):
            pass
        elif rerror == -1:
            pass
        elif self.error == -1 or math.isinf(self.error):
            self.error = rerror
        else:
            self.error = .999*self.error+.001*rerror
        return self.error
        
    def setUpperLayer(self, whichHidden = None, prevLayer = None):
        self.upper = True
        self.vis_bias = prevLayer.hid_bias[whichHidden]
        
if __name__ == '__main__':
    pass


    