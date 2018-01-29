import os
from os import rename, path
from skimage.io import imread
from glob import glob
from RandomUtils import CheckOrCreate
import sys
import numpy as np

thresh = float(sys.argv[1])

def move_img(source_file, destination):
    rename(source_file, path.join(destination, path.basename(source_file).replace('.png', 'v2.png')))

def discard(img_path, svs="0"):
    move_img(img_path, "./discard_{}".format(svs))

def valid(img_path, svs="0"):
    move_img(img_path, "./valid_{}".format(svs))

def WhiteContain(I):
    flat_in = I.reshape(224*224,3)
    def f(el):
        if (el > [200, 200, 200]).all():
            return 1
        else:
            return 0
    vec_res = map(f, flat_in)
    return np.mean(vec_res)

def OpenFileCheckMove(img_path, thresh=thresh, svs="0"):
    img = imread(img_path)[:,:,0:3]
    score = WhiteContain(img)
    if score < thresh:
        valid(img_path, svs)
    else:
        discard(img_path, svs)


files = glob('*.png')
svs_num = files[0].split('_')[1]
CheckOrCreate('./discard_{}'.format(svs_num))
CheckOrCreate('./valid_{}'.format(svs_num))

def G_OpenFileCheckMove(img_path, thresh=thresh, svs=svs_num):
    OpenFileCheckMove(img_path, thresh, svs_num)

map(G_OpenFileCheckMove, files)
