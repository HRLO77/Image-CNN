# Image-CNN

Classifies an image as dog or cat, can be used to classify different images other than those.

## Loading and encoding data
```py

# __main__.py

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
```
To load and encode data for a custom dataset run 
`data, labels = load_data('DIR', 'REGEXP', 'REGEXP')`.
For example:
`data, labels = load_data('./dataset', '^car_[0-9]+\.jpeg$', '^motorcycle_[0-9]+\.jpeg$')`
Will load images like "car_0.jpeg" labelled as 0, and "motorcycle_10.jpeg" labelled as 1 from the ./dataset directory.

## Training

The model has already been trained on 72 cat, and 72 dog images. To train on your own data, Follow steps below and train the model.
```py

# __main__.py

data, labels = load_data('./dataset', '^car_[0-9]+\.jpeg$', '^motorcycle_[0-9]+\.jpeg$', pickle=True) # add your own arguments here

model = Sequential((
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax'),
))

model.compile('nadam', loss=losses.SparseCategoricalCrossentropy(True), metrics=['acc'], jit_compile=True)

model.summary()

model.fit(data, labels, epochs=15)
```
To load the pretrained model, run `model = load_model()`.

## Making predictions

To make predictions, load or train a model (follow previous steps) and run the `predict` function with a file path to an image.

```py

# __main__.py 

def predict(fp: str, loc: bool=False, acc: float=.9, reverse: bool=False, reduce: int=1, save_prediction: bool=False):
    '''Predicts type of image, by fp provided.
    :param fp: File path to image to predict.
    :param loc: Whether or not the object should be located in the image. When locating, returns tuple[list[tuple[np.ndarray[int, int], tuple[int, int, int, int]]], tuple[int, int, int, int], tuple[int, int], tuple[int, int]] representing a list of tuples, first item being a probability as an np.ndarray, the nested 4 int tuple being the box of the zone, outer 4 int tuple being the average zone of the group foudn in the image, And the last tuple is the width and height of the image.
    :param acc: The accuracy of locating the image, as a float between 0.0001 and 1.
    :param reverse: Boolean, if True sorts by probability of group 0, otherwise sorts by probability of group 1.
    :param reduce: The factor to reduce the quality of the image by. Can result in significantly faster processing.
    :param save_prediction: Whether or not to save predictions as a pickle file.'''
    prediction = model.predict([load_image(fp)])
    
    if loc:
        img = Image.open(fp)
        img = img.convert('RGB')
        img = img.reduce(reduce)
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
        del t
        while True:
            try:
                dat += q.get_nowait()
            except Exception:
                break
        del q
        del num_threads
        del num_len
        if save_prediction:
            name = f"./dat_{datetime.datetime.utcnow().strftime('%y-%m-%d_%R-%S').replace(':', '-')}.pickle"
            open(name, 'x')
            pk.dump((dat, prediction), open(name, 'wb'))
        data: list[tuple[np.ndarray, tuple[int, int, int, int]]] = [(dat[i][0], dat[i][1]) for i in reversed(np.argsort([i[0][0][0] if reverse else i[0][0][1] for i in dat]))]
        avg  = []
        for i in data:
            if reverse:
                avg.append(i[0][0])
            else:
                avg.append(i[0][0][1])
        m = []
        for tup2 in data[:4]:
            m.append(tup2[0][0][1])
        m = sum(m) / len(m)
        largest = []
        for i in range(4):
            l=[]
            for k in data[:4]:
                l.append(k[1][i])
            largest.append(l)
        c=0
        for i in largest:
            largest[c] = sum(i) // len(i)
            c+=1
        img.close()
        return data, largest, prediction, (img.width, img.height)
    if save_prediction:
        name = f"./dat_{datetime.datetime.utcnow().strftime('%y-%m-%d_%R-%S').replace(':', '-')}.pickle"
        open(name, 'x')
        pk.dump(prediction, open(name, 'wb'))
    return prediction

print('cat_50.jpeg')
print(*predict('./images/cat_60.jpeg', True, reverse=True, acc=.95, reduce=2), sep='\n')
print('dog_50.jpeg')
print(*predict('./images/dog_60.jpeg', True, acc=.95, reduce=2), sep='\n')
```

For example, `predict('./images/test_car_image.jpeg', loc=True, acc=.96, save_prediction=True, reduce=3)`.

This returns a tuple \(when run with `loc=True`\), first item being a list of tuples, each having a np.ndarray\[np.float32, np.float32\] \(probability of groups 0 and 1 of the zone\) and a tuple\[int, int, int, int\] which is the zone that was predicted. The second item is the blend zone of the 4 most likely zones in the image. Third is a np.ndarray\[np.float32, np.float32\] being the prediction of the whole image, the last element is a tuple\[int, int\] being the width and height of the image passed (after reducing).