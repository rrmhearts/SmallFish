from RBM import RBM, Trainer
from Convolutional import Convolutional, ConvolutionalTrainer
from operator import add
from utils import *
import threading
import pickle
import sys, os
import argparse

lock = threading.Lock()

parser = argparse.ArgumentParser()
parser.add_argument("data")
args = parser.parse_args()
print args.data

# Read from argparse data
training_data = pickle.load(open(args.data, "rb"))

r = pickle.load(open("minConvPBS.p", "rb"))
t = ConvolutionalTrainer(r,.5, 0, .004)
rLayers = r.num_filters

gradients = t.calculate_gradients(training_data[0], cdk=2)
''' Training for first layer'''
for j in range(1, training_data.shape[0]):
	gradients = map(add, gradients, t.calculate_gradients(training_data[j], cdk=2) )
	
with lock:
	t.apply_gradients(*gradients, learning_rate=abs(float(r.weights.mean())) / 10000 )
	
#Save weights to file
with lock:
    print 'Saving...'
    pickle.dump(r, open("minConvPBS.p", "wb"))


