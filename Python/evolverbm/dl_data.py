#!/usr/bin/python

import os
from scipy.misc import imread
from random import random

class Data(object):
    def __init__(self, base_dir):
        self.data = []
        self.used_data = []

        self.add_base_dir(base_dir)

    def __len__(self):
        return len(self.data)

    def add_base_dir(self, base_dir):
        '''
        Add directories like this ["maledir", "femaledir"]
        '''
        #print type(base_dir)
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

    def get_random_person(self):
        # Random access into the data
        rand_idx = int(random() * len(self.data))

        if len(self.data) == 0:
            self.data = self.used_data
            self.used_data = []
        # Transfer person to used data
        person_file = self.data.pop(rand_idx)
        self.used_data.append(person_file)

        # Random access into the chip file
        im_files = [f for f in os.listdir(person_file) if f.endswith(".jpg")]
        rand_file_idx = int(random() * len(im_files))

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


    def get_random_people_set(self, num_people):
        container = []

        # Needs error testing
        for i in range(num_people):
            person, p_class = self.get_random_person()
            container.append([person, p_class])

        return container

def main():
    pass

if __name__ == '__main__':
    main()
