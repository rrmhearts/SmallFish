# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:04:15 2013

@author: Ryan
"""

class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
        
class NaNError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)