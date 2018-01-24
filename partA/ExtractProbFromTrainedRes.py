import matplotlib
matplotlib.use('agg')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from tqdm import tqdm
#import xgboost as xgb
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.models import load_model
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
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(16,16))
fig.suptitle('Visualizing partB results for partA', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)



trained_weights = sys.argv[2]

if len(sys.argv) > 3:
    mean = np.load(sys.argv[3])
else:
    mean = np.zeros(shape=3, dtype='float')

n_classes = 4
#FACTORS = [0.1]
FACTORS = [1.] # , 0.75, 0.5, 0.25, 0.1]

def vec_color(prob):
    val = np.argmax(prob)
    if val == 0:
        return [255, 255, 255]
    elif val == 1:
        return [255, 0, 0]
    elif val == 2:
        return [0, 255, 0]
    elif val == 3:
        return [0, 0, 255]
def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for x in xrange(0, image.shape[0] - windowSize[0] + stepSize, stepSize):
        for y in xrange(0, image.shape[1] - windowSize[1] + stepSize, stepSize):
            # yield the current window
            print x, y
            res_img = image[x:x + windowSize[0], y:y + windowSize[0]]
            change = False
            if res_img.shape[0] != windowSize[0]:
                x = image.shape[0] - windowSize[0]
                change = True
            if res_img.shape[1] != windowSize[1]:
                y = image.shape[1] - windowSize[1]
                change = True
            if change:
                res_img = image[x:x + windowSize[0], y:y + windowSize[1]]
                print "Changed:", x, y
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)


random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

lbl = ["Normal", "Benign", "Invasive", "InSitu"]

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in lbl])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

base_model = load_model(trained_weights)
model_prob = base_model
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
    if fact == 0.1:
        img_scale = resize(image, (224,224))
        img_scale = img_as_ubyte(img_scale)
    elif fact != 1.:
        img_scale = rescale(image, fact)
        img_scale = img_as_ubyte(img_scale)
    else:
        img_scale = image
    img_scale = img_scale.astype(float)
    # img_scale = img_scale - mean
    stepSize = 224
    windowSize = (224, 224)
    for x_b, y_b, x_e, y_e, x in sliding_window(img_scale, stepSize, windowSize):
        x = x.astype(float)
        x = np.expand_dims(x, axis=0)
    #    x = preprocess_input(x)

        features = model_prob.predict(x) 
        color = vec_color(features)
        image[x_b:x_e,y_b:(y_b+2)] = color
        image[x_b:x_e,(y_e-2):y_e] = color
        image[x_e:(x_e+2),y_b:y_e] = color
        image[(x_b-2):x_b,y_b:y_e] = color
        mid_x = (x_b + x_e) / 2 
        mid_y = (y_b + y_e) / 2 - 100
        ax.text(mid_y, mid_x, '{:10.2f}'.format(features.max()), fontsize=15)
        features_reduce =  features.squeeze()
        img_feat_list.append(features_reduce)
ax.imshow(image)
fig.axis('off')
fig.savefig(tags.replace('.tif', '_prob.tif'), bbox_inches='tight')
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
