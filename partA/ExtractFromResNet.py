import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from tqdm import tqdm
#import xgboost as xgb
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input
import scipy
from sklearn.metrics import fbeta_score
from skimage.io import imread
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
from sklearn.model_selection import StratifiedShuffleSplit
import sys
from os.path import basename
import pdb
if len(sys.argv) > 2:
    mean = np.load(sys.argv[2])
else:
    mean = np.zeros(shape=3, dtype='float')

n_classes = 4
#FACTORS = [0.25]
FACTORS = [1., 0.75, 0.5, 0.25, 0.1]

def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for y in xrange(0, image.shape[0] - windowSize[0] + stepSize, stepSize):
        for x in xrange(0, image.shape[1] - windowSize[1] + stepSize, stepSize):
            # yield the current window
            res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            change = False
            if res_img.shape[0] != windowSize[1]:
                y = image.shape[0] - windowSize[1]
                change = True
            if res_img.shape[1] != windowSize[0]:
                x = image.shape[1] - windowSize[0]
                change = True
            if change:
                res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)


random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

lbl = ["Normal", "Benign", "Invasive", "InSitu"]

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in lbl])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# use ResNet50 model extract feature from fc1 layer
base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
input = Input(shape=(224,224,3),name = 'image_input')
x = base_model(input)
x = Flatten()(x)
model = Model(inputs=input, outputs=x)

X_mat = []
y_mat = []

tags = sys.argv[1]
if basename(tags)[0] == "n":
    pre = "Normal"
elif basename(tags)[0] == "b":
    pre = "Benign"
elif basename(tags)[0:2] == "iv":
    pre = "Invasive"
elif basename(tags)[0:2] == "is":
    pre = "InSitu"

image = imread(tags).astype('uint8')
img_feat_list = []
for fact in FACTORS:
    if fact != 1.:
        img_scale = rescale(image, fact)
        img_scale = img_as_ubyte(img_scale)
    elif fact == 0.1:
        img_scale = resize(image, (224,224))
        img_scale = img_as_ubyte(img_scale)
    else:
        img_scale = image
    img_scale = img_scale.astype(float)
    img_scale = img_scale - mean
    stepSize = 224
    windowSize = (224, 224)
    for x, y, x_e, y_e, x in sliding_window(image, stepSize, windowSize):
        x = x.astype(float)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        features_reduce =  features.squeeze()
        img_feat_list.append(features_reduce)

matrix_img_feat = np.column_stack(img_feat_list)
for i in range(matrix_img_feat.shape[0]):
    matrix_img_feat[i] = np.sort(matrix_img_feat[i])


X_mat.append(matrix_img_feat.flatten())

targets = np.zeros(n_classes)
targets[label_map[pre]] = 1
y_mat.append(targets)


X = np.array(X_mat)

train_ResNet =  pd.DataFrame(X)
data = {'label': [label_map[pre]]}
y_pd = pd.DataFrame(data, columns=['label'])
train_ResNet = pd.concat([y_pd, train_ResNet], axis = 1)
p = train_ResNet.shape[1]
vec_res = train_ResNet.as_matrix().reshape(p)
vec_res_p = np.zeros(p+1, dtype='float')
vec_res_p[0] = label_map[pre]*100 + int(tags.split(".")[0][-3:]) - 1
vec_res_p[1:] = vec_res
np.save(basename(tags).replace('.tif', '.npy'), vec_res_p)
