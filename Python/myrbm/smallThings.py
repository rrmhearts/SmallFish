# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:43:02 2014

@author: Ryan
"""
import scipy
import numpy as np
from numpy import genfromtxt
my_data = genfromtxt('confmatrix.csv', delimiter=',')

#scipy.misc.imsave('confmat.jpg', my_data)
my_data = np.expand_dims(my_data, axis=0)
print my_data.shape

print "hello.."