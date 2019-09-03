#Human Not Human

One of the tricks I keep coming back to as I attempt to grow my skillset within machine learning is the right level of tutorial.  Like many things in tech, it's easy to find a hello world.  It's also easy to find the code example with some subtle nuance explained for the experts.  That middle path is what I needed.  So once I got my feet on the ground, I thought I'd write one.  Similarly with Machine Learning examples.  Setting up and understanding the data usage and structure is critical to understanding how to anser problems with ML.  And while the convenience of `mnist.load_data()` is amazing!  This doesn't answer the question of how we would functionalize ML algorithms, or all of the effort that must go into processing the input data to get to solving a problem with ML.

So I chose to start from the ground up.  My data, my curation.  Hello world examples of the image classification Convolutional Neural Net.  The goal is to understand the pipeline and processing from raw images to a working productionalized classifier.

## The data
I wanted something that was pretty easy to classify.  Something I could mint the dataset for, lastly something that I would find moderately useful.  As it happens, I recently installed a fixed position camera to a Raspberry Pi over my porch.  I figure this is a good source of relatively homogeneous images.  And bonus that I could make the motion event library stop texting me every time a cat runs by (which happens so very often).

**Note -- I have not included the dataset I used in the repo since it contains pictures of my family and my house.  I would be willing to share it with others, but want to do so by request only.

### Processing the data -- The hard work
The reality of data science as a practice is how much time is spent curating data.  The bulk of this exercise was me doing data entry.  There are cooler ways to pre train or to do unsupervised learning, but the existing datasets tend to hide how much work went into curating them.


## The model
I should be very clear that I did not develop this model.  I followed a few `hello world` binary classification tutorials until the input and output shapes matched.  I can follow the linear algebra at play to create a model like this, but I cannot yet create one on my own.  I'll include some links below for some cool primers on convolutions, and I highly recommend the [Cholet's book](https://www.manning.com/books/deep-learning-with-python) for a primer on gradient descent and the chain rule.




## Going to production
Once I had the model trained.  It is pretty trivial to save and load the model.
'''python
from keras.models import load_model
model = load_model('full.h5')
'''