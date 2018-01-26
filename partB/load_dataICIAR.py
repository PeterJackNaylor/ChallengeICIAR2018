
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

def load_ICIAR_data(mean=None):
    folder = glob(PATH)
#    folder = np.random.choice(folder, 100)
    n = len(folder)
    X = np.zeros(shape=(n, img_height, img_width, num_channels), dtype='float')
    y = np.zeros(shape=(n, num_classes), dtype='uint8')
    if mean is not None:
        mean = np.load(mean).astype('float')
        X[:] -= mean
    id_n = np.zeros(shape=n, dtype='uint8')
    print "Loading data.."
    for k, f in enumerate(folder):
        img = imread(f).astype('uint8')
        X[k] += img[:,:,0:3]
        fname = basename(f).replace('.png', '')
        lbl, svs, ind = fname.split('_')
        y[k, int(lbl)] = 1
        id_n[k] = svs
    print "Data loaded, size: {}".format(k)
    return X, y, id_n
