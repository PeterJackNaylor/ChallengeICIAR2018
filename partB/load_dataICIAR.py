
import pdb
from glob import glob
from skimage.io import imread
from os.path import basename
import numpy as np

PATH = './*.png'

num_classes = 4
num_channels = 3
img_width = 224
img_height = 224

def load_ICIAR_data():
    folder = glob(PATH)
#    folder = np.random.choice(folder, 100)
    n = len(folder)
    X = np.zeros(shape=(n, img_height, img_width, num_channels), dtype='uint8')
    y = np.zeros(shape=(n, num_classes), dtype='uint8')
    for k, f in enumerate(folder):
        img = imread(f).astype('uint8')
        X[k] = img[:,:,0:3]
        fname = basename(f).replace('.png', '')
        lbl, svs, ind = fname.split('_')
        y[k, int(lbl)] = 1
    return X, y
        
        
