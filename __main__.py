import numpy as np
import re
import os
import pickle as pk
import datetime
from PIL import Image
from keras import Sequential
from keras import layers
from keras import losses
import threading
import queue
import math

os.system('cls||clear')

def load_image(fp: str=None, img: Image.Image=None):
    '''Loads a single image.'''
    if fp != None:
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
    else:
        if img != None:
            image = img.convert('RGB')
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

# model = load_model()

def predict(fp: str, loc: bool=False, acc: float=.9):
    '''Predicts type of image, by fp provided.
    :param loc: Whether or not the object should be located in the image. When locating, returns tuple[list[tuple[np.ndarray[int, int], tuple[int, int, int, int]]], tuple[int, int], tuple[int, int]] representing a list of tuples, first item being a probability as an np.ndarray, the 4 int tuple being the box of the zone. And the last tuple is the width and height of the image.
    :param acc: The accuracy of locating the image, as a float between 0.0001 and 1.
    :param fp: File path to image to predict.'''
    if loc:
        img = Image.open(fp)
        img = img.convert('RGB')
        acc=math.ceil((1-acc-0.0001)*1000)
        q = queue.Queue(0)
        num_threads=queue.LifoQueue(0)
        def square_up(image: Image.Image, start: tuple[int, int]):
            d=[]
            for x in range(image.width)[::acc]:
                for y in range(image.height)[::acc]:
                    i = image.crop((*start, start[0]+y, start[1]+x))
                    if start[0]+y > image.height:
                        break
                    d.append((model.predict([load_image(img=i)]), (*start, start[0]+y, start[1]+x)))
                    i.close()
                if start[1]+x > image.width:
                    break
            num_threads.put_nowait(1)
            q.put_nowait(d)
        def s():
            dat = []
            m = len(range(img.width)[::acc])*len(range(img.height)[::acc])
            num_len = 0
            for x in range(img.width)[::acc]:
                for y in range(img.height)[::acc]:
                    t = threading.Thread(target=square_up, args=(img, (x,y)))
                    t.start()
            t.join()            
            while num_len != m:
                __import__('time').sleep(0.1)
                num_len = num_threads.qsize()
            while True:
                try:
                    dat += q.get_nowait()
                except Exception:
                    break
            data: list[tuple[np.ndarray, tuple[int, int, int, int]]] = [(dat[i][0], dat[i][1]) for i in reversed(np.argsort([i[0][0][1] for i in dat]))]
            # largest = [(0,0),(0,0),(0,0),(0,0)]
            # m = []
            # for tup2 in data:
            #     m.append(tup2[0][0][1])
            # # print(data)
            # m = sum(m) / len(m)
            # for tup2 in data:
            #     if tup2[0][0][1] > m:
            #         for i in range(4):
            #             # print(i%2)[0, 200, 0, 300] (==0), [100, 0, 200, 0] (==1)
            #             # if i%2 == 0:
            #                 if tup2[0][0][1] > largest[i][0] and tup2[1][i] > largest[i][1]:
            #                     largest[i] = (tup2[0][0][1],tup2[1][i])
            #             # else:
            #                 # if tup2[0][0][1] > [*largest[i].keys()][0] and tup2[1][i] > [*largest[i].values()][0]:
            #                     # largest[i] = {tup2[0][0][1]:tup2[1][i]}
            return data, model.predict([load_image(fp)]), (img.width, img.height)
        return s()  
    return model.predict([load_image(fp)])



print(predict('./images/cat_25.jpeg'))
print(*predict('./images/dog_25.jpeg', True), sep='\n')