# import tensorflow as tf
from telnetlib import SE
import numpy as np
import re
import os
import pickle as pk
import datetime
from PIL import Image
from keras import Sequential
from keras import layers
from keras import losses
# from keras.datasets import cifar10
os.system('cls||clear')

# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# print(train_images[0])
# exit()

def load_image(fp: str):
    '''Loads a single image.'''
    image = Image.open(fp)
    image = image.convert('RGB')
    image = image.resize((100, 100))    
    l=[]
    for x in range(image.width):
        p = []
        for y in range(image.height):
            p.append(image.getpixel((x,y)))
        l.append(tuple(p))
    return l

def load_data(dir: str='./images', group0_regexp: str=r'^cat_[0-9]+\.jpeg$', group1_regexp: str=r'^dog_[0-9]+\.jpeg$', pickle: bool=False):
    '''Loads multiple images from dir provided and regexp pattern for groups 0 and 1, returns data and labels'''
    d = []
    l = []
    for i in os.listdir(dir):
        if re.match(group0_regexp, i) is not None:
            d.append(load_image(os.path.join(dir, i)))
            l.append(0)
            continue
        if re.match(group1_regexp, i) is not None:
            d.append(load_image(os.path.join(dir, i)))
            l.append(1)
    d = np.array(d), np.array(l)
    
    del l
    if pickle:
        name = 'data_' + datetime.datetime.utcnow().strftime('%y-%m-%d_%R-%S').replace(':', '-') + '.pickle'
        open(name, 'x')
        pk.dump(d, open(name, 'wb'))
    return d

data, labels = load_data()

def load_model():
    '''Loads model for predictions.'''
    model = Sequential((
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax'),
))
    model.compile('nadam', loss=losses.SparseCategoricalCrossentropy(True), metrics=['acc'], jit_compile=True)
    model.load_weights('./model.h5')
    return model

model = Sequential((
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax'),
))

model.compile('nadam', loss=losses.SparseCategoricalCrossentropy(True), metrics=['acc'], jit_compile=True)

model.summary()

model.fit(data, labels, epochs=9)

model.save_weights(f"./model_{datetime.datetime.utcnow().strftime('%y-%m-%d_%R-%S').replace(':', '-')}.h5")

model = load_model()

def predict(fp: str):
    '''Predicts type of image, by fp provided.'''
    return model.predict([load_image(fp)])


print(predict('./images/cat_20.jpeg'))
print(predict('./images/dog_20.jpeg'))