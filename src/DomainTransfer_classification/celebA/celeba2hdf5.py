# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os

import h5py
import numpy as np
import scipy
from fuel.converters.celeba import convert_celeba_64

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    celeba_path = '/home/memray/Data/celeba/'
    output_path = '/home/memray/Data/celeba/'
    # convert_celeba_64(celeba_path, output_path)
    feature_data = scipy.io.loadmat('%s/celebA_tag_feats.mat' % celeba_path)
    print(len(feature_data))
