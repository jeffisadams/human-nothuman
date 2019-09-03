![alt Human Not Human](https://upload.wikimedia.org/wikipedia/commons/b/b5/Rubin2.jpg "Human Not Human")

[Rubin Vase](https://en.wikipedia.org/wiki/Rubin_vase)

#Human Not Human

## TAGS
- Keras
- Convolutional Neural Net
- Binary Classifier

One of the tricks I keep coming back to as I attempt to grow my skillset within machine learning is the right level of tutorial.  Like many things in tech, it's easy to find a hello world.  It's also easy to find the code example with some subtle nuance explained for the experts.  That middle path is what I needed.  So once I got my feet on the ground, I thought I'd write one.  Similarly with Machine Learning examples.  Setting up and understanding the data usage and structure is critical to understanding how to anser problems with ML.  And while the convenience of `mnist.load_data()` is amazing!  This doesn't answer the question of how we would functionalize algorithms, or all of the effort that must go into processing the input data to get to solving a problem.

So I chose to start from the ground up.  My data, my curation.  Hello world examples of the image classification Convolutional Neural Net.  The goal is to understand the pipeline and processing from raw images to a working productionalized classifier.

## The data
I wanted something that was pretty easy to classify.  Something I could generate the dataset for.  And lastly something that I would find moderately useful.  As it happens, I recently installed a fixed position camera to a Raspberry Pi over my porch.  I figure this is a good source of relatively homogeneous images.  And bonus that I could make the motion library event script stop texting me every time a cat runs by (which happens so very often).

**Note -- I have not included the dataset I used in the repo since it contains pictures of my family and my house.  I would be willing to share it with others, but want to do so by request only.

### Processing the data -- The hard work
The bulk of this exercise was me doing data entry.  There are cooler ways to pre-train or to do unsupervised learning, but the existing datasets tend to hide how much work went into curating them.  And to solve problems customized to a specific dataset requires usage of that specific dataset which often 

### Building training and validation datasets
The raw data is just a csv with an image path and a few columns of classified data.  I counted cars and humans for future use.  In order to input model for training and validation, I first created the input vectors and the output classifications as a list.  Then I used the sklearn train_test_split to randomly generate cohorts for train and validation data.

```python
dataset = np.ndarray(shape=(len(data), image_height, image_width, channels), dtype=np.float32)
y_dataset = []
i=0

# Set of markers so I can create an lst
files = []
for index, row in data.iterrows():
    y_dataset.append(row.human)
    img = load_img(basePath + '/' + row.filename)
    files.append('rawdata/' + row.filename)
    x = img_to_array(img)
    x = x / 255.0
    dataset[i] = x
    i += 1

x_train, x_val, y_train, y_val = train_test_split(dataset, y_dataset, test_size=0.2)
```

## The model
// TODO: Rewrite this section

I should be very clear that I did not develop this model.  I followed a few 'hello world' binary classification tutorials until the input and output shapes matched.  I can follow the linear algebra at play to create a model like this, but I cannot yet create one on my own.  I'll include some links below for some cool primers on convolutions, and I highly recommend the [Cholet's book](https://www.manning.com/books/deep-learning-with-python) for a primer on gradient descent and the chain rule.

```python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_train[1,:].shape))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```


### Why this model works relatively well
Data science purists will likely notice this is not generalized as a classifier for humans.

## Training
The first time I ran this, I trained on my laptop.  It took multiple hours, and I learned researching in the background that there are GPU accelerated notebooks for use for free from kaggle.  The second time it took only `TODO://`____ minutes.  I cannot stress enough that you should use a GPU accelerated instance where possible.

```python
batch_size=16
epochs=4

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val,y_val))
model.save('current_full.h5')
```

## Model evaluation
I ran an additional set of images as test images through the output to see how well the classifier worked.

## Going to production
Once I had the model trained.  It is pretty trivial to save and load using Keras.  The model is not a small file, but it allows for a simple python script to take in an url argument to an image, run the binary classification model on it and return the probability it is a human.  Here is the full operational classifier script.

```python
#!/usr/bin/env python
# coding: utf-8
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

import sys
from urllib.request import urlopen

import numpy as np

model = load_model('full.h5')

url = sys.argv[1]
print(url)

img = load_img(urlopen(url))
x = img_to_array(img)
x = x / 255.0

size = img.size
channels=3

dataset = np.ndarray(shape=(1, size[1], size[0], channels),dtype=np.float32)
dataset[0] = x
result = model.predict(dataset)

print(result[0][0])
```

I still have to attach this to the camera so for now I'm still getting cat notifications on my phone.  But I constantly am running `python guess.py http://...` with enthusiasm for the ground up solution.

## Links
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
    - I cannot recommend this book enough.
- [Image Kernel Visualization](http://setosa.io/ev/image-kernels/)
