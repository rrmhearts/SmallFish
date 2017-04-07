# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:42:13 2013

@author: Ryan
"""

#from numpy import exp
#from numpy.random import rand #import numpy.random as rng
#from numpy import array
import numpy as np
import random as rnd #from random import gauss
from scipy.optimize import leastsq
import scipy.signal
convolve = scipy.signal.convolve

def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))

def identity(eta):
    return eta

def bernoulli(p):
    #print 'Bernoulli: ', p
    if np.isnan(p).any():
        print p
#        raise NaNError(p)
    if type(p) is not np.ndarray:
        p = np.array(p)
    return np.random.rand(*p.shape) < p

def gaussian(mu):
    return rnd.gauss(mu, .01)
    
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow, make values  in [0, 1]
    if e.ndim == 1: # if an array
        return e / np.sum(e, axis=0) # normalizes by partition function
    else: # if a matrix
        return e / np.array([np.sum(e, axis=None)]).T  # ndim = 2
        
def prevent_overflow(x):
    maxvalue = x.max()
    dtype = type(maxvalue)
    larg_pos_val = 0.70*np.log(np.finfo(dtype).max)
    if maxvalue > larg_pos_val:
        x[x > larg_pos_val] = larg_pos_val
    return x

def gaussian_kernel(g, size):
    return np.array( [ [ g(i,j) for j in xrange(size)] for i in xrange(size)] )
    
def gaussianf(mean, center_x, center_y, sigma_x, sigma_y, rotation):
        """Returns a gaussian function with the given parameters"""
        sigma_x = float(sigma_x)
        sigma_y = float(sigma_y)
 
        rotation = np.deg2rad(rotation)

        a = np.cos(rotation)**2/2/sigma_x**2 + np.sin(rotation)**2/2/sigma_y**2;
        b = -np.sin(2*rotation)/4/sigma_x**2 + np.sin(2*rotation)/4/sigma_y**2 ;
        c = np.sin(rotation)**2/2/sigma_x**2 + np.cos(rotation)**2/2/sigma_y**2;

        def rotgauss(x,y):

            g = mean*np.exp(
                -(a * (x - center_x)**2 +
                2*b * (x-center_x) * (y-center_y) +
                  c * (y - center_y)**2) )
            return g
        return rotgauss

def moments(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        o_min = np.min(data)
        data = data - o_min
        idata = np.exp(data)
        total = idata.sum()
        X, Y = np.indices(data.shape)
        x = (X*idata).sum()/total
        y = (Y*idata).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max() #+ o_min
        return height, x, y, width_x, width_y, 0.0
 
    
def fitgaussian(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = moments(data)
        
        errorfunction = lambda p: np.ravel(gaussianf(*p)(*np.indices(data.shape)) - data)
        p, succ = leastsq(errorfunction, params)
        return (p*(succ*6)+params)/(succ*6+1)
        