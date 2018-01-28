from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from keras.utils import Sequence
from glob import glob
from os.path import join, basename
import pdb
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

PATH = './valid_*/*.png'

num_classes = 4
num_channels = 3
img_width = 224
img_height = 224


class ICIARSequence(Sequence):

    def __init__(self, path, patient, num_classes, batch_size, mean=None):
        self.path = path
        self.k = patient
        self.batch_size = batch_size
        self.num_classes = num_classes
        if mean is not None:
            self.mean = np.load(mean).astype('float')
        else:
            self.mean = None

        self._init_folder()
        self._init_random_generator()

    def _init_folder(self):

        pre_FOLD = [join(self.path, "valid_{}/*.png".format(k)) for k in range(1,11) if k != self.k]
        ALL_FOLD = []
        for sub in pre_FOLD:
            ALL_FOLD += glob(sub)
        ALL_FOLD = np.array(ALL_FOLD)

        def cut_name(name):
            return int(basename(name).split('_')[0])

        def cut_name_assign(name):
            c = int(basename(name).split('_')[0])
            arr = np.zeros(self.num_classes, dtype="uint8")
            arr[c] = 1
            return arr

        y_label = map(cut_name, ALL_FOLD)
        y_onehot = map(cut_name_assign, ALL_FOLD)
        self.n = len(ALL_FOLD)

        shuffle_index = list(range(self.n))
        np.random.shuffle(shuffle_index)
        self.folder = ALL_FOLD[shuffle_index]

        self.y = np.array(y_label)[shuffle_index]
        self.y_onehot = np.array(y_onehot)[shuffle_index]

    def _init_random_generator(self):
        datagen_args = dict(rotation_range=180,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0,
                    zoom_range=0,
                    fill_mode='reflect',
                    horizontal_flip=True,
                    vertical_flip=True)
        self.datagen = ImageDataGenerator(**datagen_args)

    def __len__(self):
        return self.n / self.batch_size

    def __getitem__(self, idx):
        batch_name = self.folder[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_onehot = self.y_onehot[idx * self.batch_size:(idx + 1) * self.batch_size]

        def f(name):
            return self.datagen.random_transform(imread(file_name).astype('uint8')[:,:,0:3])

        batch_x = np.array([f(file_name) for file_name in batch_name]).astype(float)
        batch_y = np.array(batch_y)
        if self.mean is not None:
            batch_x[:] -= self.mean
        return batch_x, batch_y_onehot
    def weight_dic(self):
        dic = class_weight.compute_class_weight('balanced', np.unique(self.y), self.y)
        return dic

class ICIARSequenceTest(ICIARSequence):
    def _init_folder(self):

        ALL_FOLD = glob(join(self.path, "valid_{}/*.png").format(self.k))
        ALL_FOLD = np.array(ALL_FOLD)

        def cut_name(name):
            return int(basename(name).split('_')[0])

        def cut_name_assign(name):
            c = int(basename(name).split('_')[0])
            arr = np.zeros(self.num_classes, dtype="uint8")
            arr[c] = 1
            return arr

        y_label = map(cut_name, ALL_FOLD)
        y_onehot = map(cut_name_assign, ALL_FOLD)

        self.n = len(ALL_FOLD)

        self.folder = ALL_FOLD
        self.y = np.array(y_label)
        self.y_onehot = np.array(y_onehot)

    def _init_random_generator(self):
        print "no random generator for test, Not implemented"
    def __getitem__(self, idx):
        batch_name = self.folder[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_onehot = self.y_onehot[idx * self.batch_size:(idx + 1) * self.batch_size]

        def f(name):
            return imread(file_name).astype('uint8')[:,:,0:3]

        batch_x = np.array([f(file_name) for file_name in batch_name]).astype(float)
        batch_y = np.array(batch_y)
        if self.mean is not None:
            batch_x[:] -= self.mean
        return batch_x, batch_y_onehot
