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
avgRE = r.get_avg_error(training_data[0])

print 'Error: ', avgRE
