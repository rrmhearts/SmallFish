#!/usr/bin/python

__author__ = "H. Nathan Rude"

import sys
import os
from scipy.misc import imread

# Adds the parent directory to the system path to allow access to algorithms
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from lib import weka, timer
from lib.algorithm import hog, lbp, lhsv, gabor
from lib.datastorage import Data

# File and directory stuff
#zoom=["Zoom", "NoZoom"]
zoom_level=["Zoom"]  # Zoom and/or NoZoom
orientation=["F", "B"] # F and/or B
base_dir = "C:\\Images\\SWAG"
male_dirs = [base_dir + "\\male\\" + z + '\\' + o for z in zoom_level for o in orientation]
female_dirs = [base_dir + "\\female\\" + z + '\\' + o for z in zoom_level for o in orientation]
# Make sure the files are images by comparing the extentions to a list of valid extensions
valid_ext = [".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".gif", ".tiff", ".tif"]
chip_limit = 10
# Get files
#male_files = get_files(male_dirs, valid_ext, gender="male.truth")
#female_files = get_files(female_dirs, valid_ext, gender="female.truth")
print "Adding male files"
male_files = Data(male_dirs)
print "Adding female files"
female_files = Data(female_dirs)
male_count = len(male_files)
female_count = len(female_files)
all_files = male_files + female_files
del male_files
del female_files

# Creation of output directory
output_dir = "./output/weka/swag/"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Create weka file name by joining the algorithm names that are involved
weka_file = output_dir + "swag_{0}_{1}_{2}_{3}.arff".format("M-" + str(male_count)
    + "-F-" + str(female_count), weka_name, '-'.join(zoom_level), '-'.join(orientation))

# If the name exists, append numbers to the end of the filename until it is unique
counter = 1
tmp_file = weka_file
while os.path.exists(tmp_file):
    split_name = weka_file.split('.')
    split_name[1] = split_name[1] + str(counter)  # Skip the './'
    tmp_file = ".".join(split_name)
    counter += 1
weka_file = tmp_file

weka = weka.Weka(relation="Gender")

# Print stats
print "Number of training images: %s" % (male_count + female_count)
print "\tMale: %s" % male_count
print "\tFemale: %s" % female_count

# Add the classes and save the file
weka.add_attribute("class", "{male, female}")
weka.write_file(weka_file)