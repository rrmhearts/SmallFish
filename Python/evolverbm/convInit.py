import numpy as np
import pickle, os
from RBM import RBM, Trainer
from Convolutional import Convolutional, ConvolutionalTrainer
from mnist import MNIST

print 'Test ConvRBM Init'
rLayers = 40
'''Conv(num_filters, filter_shape, pool_shape, binary=True, scale=0.001):'''
r = Convolutional(rLayers, [12,12], [2,2], False) #Convolutional(2, [3, 3], [2, 2])
#r = pickle.load(open("minConvLayer1.p", "rb"))
pickle.dump(r, open("minConvPBS.p", "wb") )

mdir = "MNISTSet"
if not os.path.exists(mdir):
	os.makedirs(mdir)
	
mn = MNIST()    
training_data, classes = mn.load_training()
training_data = np.asarray(training_data)
training_data = np.reshape(training_data, [len(training_data),28,28])

# Write MNIST to files
counter = 0
np.random.shuffle(training_data)
for i in range(0, len(training_data), 30):
	endi = min(i+10, len(training_data))
	pickle.dump(training_data[i:endi], open(os.path.join(mdir, "mnist"+str(counter)+".p"), "wb"))
	counter = counter + 1

'''
for i in range(0, len(td), 10000):
	endi = min(i+10, len(td)-1)
	pickle.dump(td[i, endi], open("mnist"+str(counter)+".p", "wb"))
	counter = counter + 1
	'''