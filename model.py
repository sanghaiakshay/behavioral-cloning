#import all the required modules
import pickle
from keras.preprocessing import image
import numpy as np 
from keras.applications.imagenet_utils import preprocess_input	
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import image, sequence
from keras.layers import Embedding, Input, Cropping2D, Lambda, Conv2D, Dropout
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, Activation, Flatten
from keras.layers import BatchNormalization, Concatenate
from keras.models import Model
from keras.optimizers import Adam
import math
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import keras.applications as ka
import os
import csv
from sklearn.utils import shuffle
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from keras.preprocessing import image

batch_size = 128
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
#Remove the 1st one as it contains the header
samples = samples[1:]
#Use train_test_split to create train and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Define train generator function. We shuffle the samples after each epoch.  
def tr_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                #Use center image
                center_image = Image.open(center_name)
                center_image = center_image.resize((160,160))
                center_angle = float(batch_sample[3])               
                #Use left image with correction factor
                left_image = Image.open(left_name)
                left_image = left_image.resize((160,160))
                left_angle = center_angle + 0.2
                #Use right image with correction factor              
                right_image = Image.open(right_name)
                right_image = right_image.resize((160,160))
                right_angle = center_angle - 0.2
                #Flip the center image, since we have few samples for right curve                
                flip_image = center_image.transpose(Image.FLIP_LEFT_RIGHT)
                flip_angle = center_angle*-1
                
                center_image = np.array(center_image)
                left_image = np.array(left_image)
                right_image = np.array(right_image)
                flip_image = np.array(flip_image)
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                images.append(flip_image)
                angles.append(flip_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

#Define validation generator.            
def val_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = Image.open(name)
                center_image = center_image.resize((160,160))
                center_angle = float(batch_sample[3])
                center_image = np.array(center_image)
                images.append(center_image)
                angles.append(center_angle)

            X_val = np.array(images)            
            y_val = np.array(angles)
            yield shuffle(X_val, y_val)

train_generator = tr_generator(train_samples, batch_size=batch_size)
validation_generator = val_generator(validation_samples, batch_size=batch_size)

#Define model. Model does the preprocessing of images (cropping and normalizing)
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,160,3)))
model.add(Lambda(lambda x:  (x / 255.0) - 0.5))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
#Default LR for adam is 0.001 which is working fine. 
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size,
                    validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, epochs=30, verbose = 1)
#Save the model
model.save('model.h5')
