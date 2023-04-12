import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

train_dir = 'C:/Users/isaac/Documents/datasets/fruitveggie/train'
test_dir = 'C:/Users/isaac/Documents/datasets/fruitveggie/test'
validation_dir = 'C:/Users/isaac/Documents/datasets/fruitveggie/validation'

batch_size = 32
image_height = 160
image_width = 160

train_datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (image_height, image_width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

validation_gen = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (image_height, image_width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

model = kb.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(image_height, image_width, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(512, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(36, activation = 'softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_gen,
    steps_per_epoch = train_gen.n,
    epochs = 100,
    validation_data = validation_gen,
    validation_steps = validation_gen.n)