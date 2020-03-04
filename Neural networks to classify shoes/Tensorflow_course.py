# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:25:30 2020

@author: fgm.si

Keras course from TensorFlow Youtube
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

##################################################
### Part 1&2: Trainning a NN to identify shoes ###
##################################################

# Data 
fashion_mnist = keras.datasets.fashion_mnist

# Train data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizing images
train_images = train_images / 255
test_images = test_images / 255

# Model
model = keras.Sequential([
        keras.layers.Flatten(), # Size of the images
        keras.layers.Dense(128, activation = tf.nn.relu), # 128 ReLU functions
        keras.layers.Dense(10, activation = tf.nn.softmax) # Size of the number of items of clothing in the dataset that are shoes
        ]) # Softmax takes the higher probability item of the 10th and assign the prob of 1 and the rest 0.

model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 5) # The more epochs, the more accuracy but risk of overfitting

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[2]) # Gives the probability of the item to be one of the 10 categories
# In this case the item is categorize inside the 2nd category with 99% prob
print(test_labels[2]) # Indeed, is in the second categorie 


##############################################################
### Part3: Convotional NN to identify features in an image ###
##############################################################

"""
Instead of whole images, the model can learn from the parts
of the images. This comes the "Convolutional" part. We put a 
convolutional layer on top of the NN. We have not selected 
the shape of the images previously
        
In this case, it creates 64 filters and multiply each of them 
acoss the image. Then, each epoch it will figure it out which 
filters gives the best signals that help match better the 
images with  their labels.
"""
# Model

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu",
                               input_shape = (28,28,1)),
        tf.keras.layers.MaxPoolings2D(2,2)
        tf.keras.layers.Flatten(),
        tf.keras.Dense(128, activation = tf.nn.relu),
        tf.keras.Dense(10, activation = tf.nn.softmax)
        ])

# We load the image i, that at the end is a 512x512 matrix
i = misc.ascent()
plt.grid(False)
plt.gray()
plt.axis("off")
plt.imshow(i)

# Store in an array
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# We create a filter
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# Iterate over the image, leaving a 1 pixel margin, and multiply
# out each of the neighbors of the current pixel by the value 
# defined in the filter.

for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (i[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (i[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (i[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (i[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (i[x, y] * filter[1][1])
      output_pixel = output_pixel + (i[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (i[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (i[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (i[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel # * weight
      if(output_pixel<0):
        output_pixel=0
      if(output_pixel>255):
        output_pixel=255
      i_transformed[x, y] = output_pixel

# Image filtered
      
plt.grid(False)
plt.gray()
plt.axis("off")
plt.imshow(i_transformed)

"""
Using [-1,0,1,-2,0,2,-1,0,1] gives us a very strong set of vertical lines.
If we change the size (e.g: 5x5) or the filter (e.g: [-1,-2,-1,0,0,0,1,2,1]),
the output pixel matrix would change, as the neighbour pixels change.

Why we put a filter in the first place?

The goal is to reduce the overall amount of information in an image while 
maintaining the features.
"""

# Pooling

new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    pixels.sort(reverse=True)
    newImage[int(x/2),int(y/2)] = pixels[0]
 
# With a (2,2) pooling, we have reduced the dimensaion to a 256x256 image
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.axis('off')
plt.show()

################################################################
### Part4: Build an image classifier for rock,paper,scissors ###
################################################################

https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb
