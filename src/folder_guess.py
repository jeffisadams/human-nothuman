#!/usr/bin/env python
# coding: utf-8

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

import os
from shutil import copyfile, rmtree
import sys

import numpy as np


def isHuman(filepath):
    img = load_img(filepath, target_size=(target_height, target_width))
    x = img_to_array(img)
    x = x / 255.0

    size = img.size

    dataset = np.ndarray(shape=(1, size[1], size[0], channels),dtype=np.float32)
    dataset[0] = x
    result = model.predict(dataset)
    return result[0][0]

# load the model
model = load_model('../models/human_not_human.h5')

# Base values
target_height = 180
target_width = 320
channels = 3

accept_threshold = 0.8

path = sys.argv[1]
print(path)

if not os.path.exists(path):
    raise Exception('`{}` does not exist or is not a directory'.format(path))


# Initialize the directory and ensure it's only our current version
dstPath = './data/tested/'
if os.path.exists(dstPath):
    rmtree(dstPath)
os.makedirs(dstPath)
os.makedirs(os.path.join(dstPath, '0'))
os.makedirs(os.path.join(dstPath, '1'))
for f in os.listdir(path):
    if f.endswith('.jpg'):
        filepath = os.path.join(path, f)
        human = '1' if isHuman(filepath) > accept_threshold else '0'
        copyfile(filepath, os.path.join(dstPath, human, f))