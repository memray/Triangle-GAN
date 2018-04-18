# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import json
from scipy.io import loadmat
import numpy as np
from collections import Counter

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    feature_data = loadmat('/home/memray/Data/coco/zhegan/tag_feats.mat')
    print(len(feature_data))

    dataset_json = json.load(open('/home/memray/Data/coco/zhegan/dataset.json', 'r'))
    print(len(dataset_json))

    counter = Counter(np.concatenate([np.concatenate([s['tokens'] for s in img['sentences']]) for img in dataset_json['images'] if
                            img['split'] == 'train']))

    sorted_words = sorted(counter.items(), key=lambda x:x[1], reverse=True)
    for id, (k,v) in enumerate(sorted_words):
        print('%d \t %s:%s' % (id, k, v))
