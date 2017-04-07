# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:43:02 2014

@author: Ryan
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq

#from numpy import genfromtxt
#my_data = genfromtxt('confmatrix.csv', delimiter=',')

#scipy.misc.imsave('confmat.jpg', my_data)
#my_data = np.expand_dims(my_data, axis=0)
#print my_data.shape

print "hello.."

def gaussian_simple(mean, center_x, center_y, sigma_x, sigma_y, rotation):
        """Returns a gaussian function with the given parameters"""
        sigma_x = float(sigma_x)
        sigma_y = float(sigma_y)
 
        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)
        
        print "center"
        print center_x
        print center_y
        def rotgauss(x,y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = mean*np.exp(
                -((abs(center_x-xp)/sigma_x)**2+
                  (abs(center_y-yp)/sigma_y)**2)/2.)
            return g
        return rotgauss
        
def gaussian(mean, center_x, center_y, sigma_x, sigma_y, rotation):
        """Returns a gaussian function with the given parameters"""
        sigma_x = float(sigma_x)
        sigma_y = float(sigma_y)
 
        rotation = np.deg2rad(rotation)

        a = np.cos(rotation)**2/2/sigma_x**2 + np.sin(rotation)**2/2/sigma_y**2;
        b = -np.sin(2*rotation)/4/sigma_x**2 + np.sin(2*rotation)/4/sigma_y**2 ;
        c = np.sin(rotation)**2/2/sigma_x**2 + np.cos(rotation)**2/2/sigma_y**2;
        print "center"
        print center_x
        print center_y
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
        
        errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
        p, succ = leastsq(errorfunction, params)
        return (p*(succ*6)+params)/(succ*6+1)
        

 
#arr = np.ndarray.ndim((5,5))
    
kern = gaussian ( .5, 0, 0, 3, 1, 45)
k2 = gaussian ( .1, 6, 6, 1, 1, 40)

print kern(0,0)
print kern(1,2)
print kern(2,4)
print kern(5,5)
print kern(15,15)
#
#size = 10
#x, y = np.mgrid[-size:size+1, -size:size+1]
#g = k2(x,y) -kern(x,y)
#
#plt.imshow(g, cmap=plt.get_cmap('jet'), interpolation='nearest')
#plt.colorbar()
#plt.show()
#
def gaussian_kernel(g, size):
    return np.array( [ [ g(i,j) for j in xrange(size)] for i in xrange(size)] )
#    return np.array( [ [ (i-size*1.0/2)**2+(j-size*1.0/2)**2 for j in xrange(size)] for i in xrange(size)] )
#
g = gaussian_kernel(k2, 10)# - gaussian_kernel(kern, 10)
p = fitgaussian(g)
print p

p = fitgaussian(-g)
print p
print g
plt.imshow( g)


#import matplotlib as mpl
#mpl.use( "qt4" )
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
#import numpy as np
#
#print plt.get_backend()
#
#im = np.arange(100)
#im.shape = 10, 10
#
#fig = plt.figure(1, (50., 50.))
#grid = ImageGrid(fig, 111, # similar to subplot(111)
#                nrows_ncols = (30, 20), # creates 2x2 grid of axes
#                axes_pad=0.1, # pad between axes in inch.
#                )
#print "Halfway..."
#for i in range(600):
#    grid[i].imshow(im) # The AxesGrid object work as a list of axes.
#print "before show"
##plt.show()
#
##plt.savefig("hello.png")
#fig.savefig("Hello.png")

print "fini."