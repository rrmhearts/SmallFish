# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 11:20:12 2014

@author: Ryan
"""

import random, numpy
import csv

    
if __name__ == '__main__':
    #cProfile.run('testConvolution()')
    #testConvolution()
#    count = 0
#    for i in range(100000):
#        if random.uniform(0,1) < .5 or random.uniform(0,1) < .3 or random.uniform(0,1) < .3 or\
#            random.uniform(0,1) < .3:
#                count += 1.
#    print str(count/100000)
                


    with open('eggs.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
        spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])





    with open('eggs.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([1,2,3,4,5])
        spamwriter.writerow([6,7,8,9,10])



    