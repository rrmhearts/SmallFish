import pkg_resources, sys
from mnist import MNIST
from scipy.misc import imsave
from RBMPlay import RBM, Trainer
from conv_playground import Convolutional
import numpy as np
import pickle
import sys, os, time

print sys.version

print pkg_resources.get_distribution("matplotlib").version
saveType = "Official"
r = pickle.load(open("minConvLayer"+saveType+".p", "rb"))

data = np.random.randn(28,28)+.8 #training_data[0]

for i in range(30):
	minRecon = r.reconstruct(data, 2)
	imsave(os.path.join("Images", "reconL_"+str(i)+"_0n.jpg"), data)
	imsave(os.path.join("Images", "reconL_"+str(i)+"_1n.jpg"), minRecon)
	data = minRecon