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

The model has already been trained on 21 cat, and 21 dog images. To train on your own data, Follow steps below and train the model.
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
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax'),
))

model.compile('nadam', loss=losses.SparseCategoricalCrossentropy(True), metrics=['acc'], jit_compile=True)

model.summary()

model.fit(data, labels, epochs=9)
```
To load the pretrained model, run `model = load_model()`.

## Making predictions

To make predictions, load or train a model (follow previous steps) and run the `predict` function with a file path to an image.

```py

# __main__.py 

def predict(fp: str):
    '''Predicts type of image, by fp provided.'''
    return model.predict([load_image(fp)])


print(predict('./images/cat_20.jpeg'))
print(predict('./images/dog_20.jpeg'))
```

For example, `predict('./images/test_car_image.jpeg')`.
