# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:59:23 2013

@author: Ryan
"""
from RBM import RBM
from utils import *
import numpy as np
import cPickle as pickle
from PIL import Image
from scipy.misc import imsave
import weka
from datastorage import Data
from Convolutional import getData
from Convolutional import Convolutional
rLayers, last_percent = 20, -1
print 'Getting data..'

tort = "TRAINING"
#tort = "TESTING"

testing_data, classes =  getData(["C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\male\\Zoom", "C:\\Users\\Ryan\\Documents\\\SWAG_"+tort+"\\female\\Zoom"])

print 'Loading r, r2...'
#r = pickle.load(open("minConvLayer1.p", "rb"))
#r2 = pickle.load(open("minConvLayer2.p", "rb"))
rLayers = 30#r.num_filters
r = Convolutional(rLayers, [20,12], [2,2], False, .1) #Convolutional(2, [3, 3], [2, 2])
r2 = Convolutional(rLayers, [22,14], [2,2], True, .1)
r3 = Convolutional(rLayers, [23,14], [2,2], True, .1)

#r=pickle.load(open("randomLayer1.p", "rb"))
#r2=pickle.load(open("randomLayer2.p", "rb"))
#r3=pickle.load(open("randomLayer3.p", "rb"))
#pickle.dump(r, open("randomLayer1.p", "wb"))
#pickle.dump(r2, open("randomLayer2.p", "wb"))
#pickle.dump(r3, open("randomLayer3.p", "wb"))
rLayers = r.num_filters


print 'Number of Layers: ', rLayers
weka_file = tort+"_random_3L_30F_output.arff"
wk = weka.Weka(relation="Gender")
flag = True

print 'Writing Weka file: ', weka_file
for i in range(len(classes)):
    out1 = r.pooled_expectation(testing_data[i] )#hidden_max_pooling(testing_data[i] )
    out2 = np.array([r2.pooled_expectation(out1[k]) for k in range(rLayers)])
    #print 'out1 size: ', out2.shape
    print out2.shape
    features = np.array([[r3.pooled_expectation(out2[l,k]) for k in range(rLayers)] for l in range(rLayers)])
    #print 'out feat: ', features.shape    
    #imsave(['im',str(i)],features[0])
    #features = features.ravel()
    #if i < 3:
    #    print out2#features
    features = features.ravel()
    gender = classes[i]
    if flag:            
        f_shape = features.shape[0]
        flag = False
        for j in range(f_shape):
            wk.add_attribute('dl'+ str((j+1)), 'NUMERIC')
        print 'Feature length: ', features.shape[0]
    bob = ''
    for j in features:
        bob += str(j)+','
    wk.add_data(bob + gender.lower())
    
    percent = int(100*float((i+1))/len(classes))
    if percent > last_percent:
        print 'Percent through samples: ', str(percent), '%'
    last_percent = percent
    
wk.add_attribute('class', "{male, female}")
wk.write_file(weka_file)
print 'Finished!'