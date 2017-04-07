#!/usr/bin/python

import os
import cPickle as pickle
import numpy as np
from PIL import Image
from scipy.misc import imsave
from MyError import NaNError, MyError
from scipy.misc import imread
from random import random

def getData(loc = ["C:\\Users\\Ryan\\Documents\\SWAG_SUBSET\\male\\Zoom", "C:\\Users\\Ryan\\Documents\\SWAG_SUBSET\\female\\Zoom"]):
    d = Data(loc)
    input_data = d.get_people_set()
    #print input_data[0][0].shape
    #print input_data[0][1]
    #print input_data[0][1][1]
    
    ''' Remove images that are not of the right size'''
    for subject in input_data:
        pic = subject[0]#input_data[i][0]
        if pic.shape != (150,90,3):
            print 'Removed an element'
            input_data.remove(subject)#input_data[i]) #this broke?alueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            print 'New size: ', len(input_data)

    ''' Pull correct images into data array'''
    data = np.zeros((len(input_data), input_data[0][0].shape[0],input_data[0][0].shape[1]))
    for i in range(len(input_data)):
        pic = input_data[i][0]
        #print type(pic), ' ', pic.shape
        if pic.shape == (150,90,3):
            #print 'success!'
            data[i,:,:] = Image.fromarray(pic, 'RGB').convert('L')
        else:
            raise MyError(pic.shape)
            #data[i,:,:] = data[i-1,:,:]
    ''' Get classes from data'''
    classes = [subject[1] for subject in input_data]
    #print classes
    pickle.dump([data, classes], open("SWAG_data.p", "wb"))
    return data/255, classes
    
class Data(object):
    def __init__(self, base_dir=[]):
        self.data = []
        self.used_data = []

        if base_dir is not []:
            self.add_base_dir(base_dir)

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        if type(other) is Data:
            d = Data()
            d.data = self.data + other.data
            return d
        else:
            raise ValueError, "Can't concatenate %s with %s" %(type(self), type(other))


    def add_base_dir(self, base_dir):
        '''
        Add directories like this ["maledir", "femaledir"]
        '''
        # Convert string input to list
        if type(base_dir) is str:
            base_dir = [base_dir]

        # Loop through the lists of base directories
        for base in base_dir:
            # Load up the data var with the file paths of the people directory
            for root, dirs, files in os.walk(base):
                # Make sure we are in the "chips" directory
                if dirs == [] and 'chips' in root:
                    self.data.append(root)

    def get_person(self):
        # Random access into the data
        #rand_idx = int(random() * len(self.data))

        # Transfer person to used data
        #person_file = self.data.pop(rand_idx)

        if len(self.data) is 0:
            raise IndexError, "There are no more people left for data"

        person_file = self.data.pop(0)
        self.used_data.append(person_file)

        # Random access into the chip file
        im_files = [f for f in os.listdir(person_file) if f.endswith(".jpg")]
        print person_file
        rand_file_idx = int(random() * len(im_files)) #this will be zero

        # Read in a random image in the chips directory
        p_data = imread(os.path.join(person_file, im_files[rand_file_idx]))

        # Find class based off of file path. Female and male label is in the file path
        file_path = person_file.split('\\')
        if "female" in file_path:
            p_class = "female"
        elif "male" in file_path:
            p_class = "male"
        else:
            p_class = "unknown"

        return p_data, p_class


    def get_people_set(self, num_people=None):
        container = []
        
        if num_people is None:
            num_people = len(self.data)
        for _ in range(num_people):
            try:
                person, p_class = self.get_person()
                container.append([person, p_class])
            except IndexError as e:
                print e
                break

        return container

def main():
    pass

if __name__ == '__main__':
    main()
