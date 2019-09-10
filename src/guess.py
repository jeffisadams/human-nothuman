#!/usr/bin/env python
# coding: utf-8
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

import sys
from urllib.request import urlopen

import numpy as np

# Base values
target_height = 180
target_width = 320
channels = 3

model = load_model('../models/human_not_human.h5')

url = sys.argv[1]
print(url)

img = load_img(urlopen(url), target_size=(target_height, target_width))
x = img_to_array(img)
x = x / 255.0

size = img.size
channels=3

dataset = np.ndarray(shape=(1, size[1], size[0], channels),dtype=np.float32)
dataset[0] = x
result = model.predict(dataset)

print(result[0][0])
