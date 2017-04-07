# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:40:00 2013

@author: Ryan based on Leif's RBM
"""
import logging
import numpy as np
import numpy.random as rng
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
        sample = bernoulli( self.hidden_expectation(visible))
        #sample = np.array([ (sample[i] + sample[(i+1)%len(sample)] ) % 2 for i in range(len(sample)) ])
        #sample = np.array([ (sample[i] + sample[(i-1)%len(sample)] ) % 2 for i in range(len(sample)) ])
        return sample #bernoulli( self.hidden_expectation(visible))
        
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
            exp = self.hidden_expectation(visible)
            hidden = bernoulli(exp)
            #hidden = np.array([ (hidden[i] + hidden[(i+1)%len(hidden)] ) % 2 for i in range(len(hidden)) ])
            #hidden = np.array([ (hidden[i] + hidden[(i-1)%len(hidden)] ) % 2 for i in range(len(hidden)) ])

            #assert hidden.max() == 1
            
            # Error checking
            if np.isnan(exp).any():
                print 'NaN'
                exp = np.nan_to_num(exp)
                #print 'hidden: ', hidden
                #print 'hidden size: ', hidden.shape
                raise NaNError(exp)
            yield visible, hidden #exp
            # Ryan experiment - push neighbors away from each other


            # end Ryan experiment
            visible = self.visible_expectation(hidden)#bernoulli(hidden))

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
        rvisible = self.reconstruct(visible,2)
        difference = rvisible - visible
        rerror = np.sum(difference*difference)
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
        
class Trainer(object):
    ''' Training functions for RBM
    '''

    def __init__(self, rbm, momentum=0., l2=0., target_sparsity=None):
        '''
            Trainer for traditional RBM.
            rbm: RBM Trainer is acting on
            momentum: Momentum term for updating gradients
            l2: Weight Decay term
            target_sparsity: Sparsity, forces average hidden activation to be
                close to this number
        '''
        self.rbm = rbm
        self.momentum = momentum
        self.l2 = l2 #Weight Decay term
        self.target_sparsity = target_sparsity
        self.sparseness = None
        self.learning_rate = .0002
        self.grad_weights = np.zeros(rbm.weights.shape, float)
        self.grad_vis = np.zeros(rbm.vis_bias.shape, float)
        self.grad_hid = np.zeros(rbm.hid_bias.shape, float)
        self.upper = False
        
        #if target_sparsity is None:
        #    self.target_sparsity = rbm

    def set_target_sparseness(self, sparsity):
        self.target_sparsity = sparsity
        
    def learn(self, visible, learning_rate=0.2, cdk=1):
        ''' Calculate and apply gradients
            visible: data to learn
            learning_rate: How much gradient affects the weights
            cdk: How far to take contrastive divergece (1 is quick)
        '''
        self.learning_rate = 0.99*self.learning_rate + 0.01*learning_rate
        gradients = self.calculate_gradients(visible,cdk)
        self.apply_gradients(*gradients, learning_rate=self.learning_rate)

    def calculate_gradients(self, visible_batch, cdk=1):
        '''Calculate gradients for a batch of visible data.

        Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

        visible_batch: A (batch size, visible units) array of visible data. Each
          row represents one visible data sample.
        '''
        passes = self.rbm.iter_passes(visible_batch)
        v0, h0 = passes.next()
        #v1, h1 = passes.next()
        ''' Allows user to choose number of steps for contrastive divergence'''
        for i in range(cdk):
            v1, h1 = passes.next()
            if i + 1 == cdk:
                break

        gw = (np.dot(h0.T, v0) - np.dot(h1.T, v1)) / len(visible_batch)
        
        if not self.upper:
            gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=0)
        
        # Changed to gh + self.Tar.. 0) (see paper)
        if self.target_sparsity is not None:
            self.sparseness = self.target_sparsity - h0.mean(axis=0)
        if self.sparseness is not None:
            gh += self.sparseness

        logging.debug('displacement: %.3g, hidden std: %.3g',
                      np.linalg.norm(gv), h0.std(axis=1).mean())

        return gw, gv, gh
    
    def apply_gradients(self, weights, visible, hidden, learning_rate=0.2):
        '''Apply a gradient to the named parameter array.
        
        weights: weight matrix to update
        visible: visible bias
        hidden: hidden bias
        learning_rate: amount to update (rate of convergence)
        '''
        def update(name, g, _g, l2=0):
            target = getattr(self.rbm, name)
            g *= 1 - self.momentum
            g += self.momentum * (_g - l2 * target)

            # Update RBM weights
            target += learning_rate * g

            # Update Trainer weights
            _g[:] = g
        
        if not self.upper:
            update('vis_bias', visible, self.grad_vis)
        update('hid_bias', hidden, self.grad_hid)
        update('weights', weights, self.grad_weights, self.l2)
    def add_noise(self, scale = .001):
        '''Apply a gradient to the named parameter array.
        
        weights: weight matrix to update
        visible: visible bias
        hidden: hidden bias
        learning_rate: amount to update (rate of convergence)
        '''
        def update(name):
            target = getattr(self.rbm, name)
            # Update RBM weights
            noise = np.random.randn(*target.shape)
            target += scale * noise
        
        update('vis_bias')
        update('hid_bias')
        update('weights')
    def setUpperLayer(self):
        self.upper = True
    
def testRBM():
    Z = 1
    r = RBM(6, 6, False, .001) 
    r2 = RBM(6, 5, True, .001)
  
    t = Trainer(r, .2)#,0,0,.5)
    t2 = Trainer(r2, .2)#,0,0,.5)
  
    training_data = np.array([[0.,1.,2.,3.,4.,5.],[0,1.1,2.1,3.1,4.1,4.9],[.1,.9,1.9,2.9,3.9,4.9],\
                              [.1,1.1,2.05,2.95,4.05,4.95],[0,.98,1.95,3.01,3.99,4.91]])/Z
    oldE = 20
    for i in range(10000): #while avg > curr:
        np.random.shuffle(training_data)
        t.learn(training_data, .2, cdk=10+i/900)
        #for j in range(training_data.shape[0]):
        en1 = r.get_entropy(training_data)
        avgE = r.get_avg_entropy()
        recE = r.get_avg_error(training_data)#.get_reconstruction_error(training_data)
        #avg = avg_entropy(curr, avg)
        if np.mod(i,10) == 0:
            print i, ', Entropy 1: ', en1, ', ', avgE, ', ', recE
            oldE = avgE
    for i in range(10000):
        np.random.shuffle(training_data)
        en2 = r2.get_entropy(r.hidden_sample(training_data))
        avgE = r2.get_avg_entropy()
        recE = r.get_avg_error(training_data)#.get_reconstruction_error(training_data)
        t2.learn(r.hidden_sample(training_data), .2, cdk=1+i/800)
        if np.mod(i,10) == 0:
            print 'Entropy 2: ', en2, ', ', avgE, ', ', recE
    
    print 'Final entropies: ', r.get_entropy(training_data), ', ', r2.get_entropy(r.hidden_sample(training_data))
    print 'Weights: ', r.weights, "\n", r2.weights
    math_in = np.array([0.,1.,2.,3.,4.,5.])/Z
    math2=r2.reconstruct(r.hidden_sample( math_in))
    math=r.visible_expectation(math2 )
    print 'Input: ', math_in
    print 'Reconstruct: ', math
    out1 = r.hidden_expectation(math_in)#np.array([1,.9,0,0,0,0]))
    out2 = r2.hidden_expectation( out1 )
    print 'Layer 2: ', out2
    print 'Layer 1: ', out1
    print 'Visible: ', r.visible_expectation(r2.visible_expectation(out2) )
if __name__ == '__main__':
    testRBM()
