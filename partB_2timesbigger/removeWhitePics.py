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

def discard(img_path):
    move_img(img_path, "./discard")

def valid(img_path):
    move_img(img_path, "./valid")

def WhiteContain(I):
    flat_in = I.reshape(224*224,3)
    def f(el):
        if (el > [200, 200, 200]).all():
            return 1
        else:
            return 0
    vec_res = map(f, flat_in)
    return np.mean(vec_res)

def OpenFileCheckMove(img_path, thresh=thresh):
    img = imread(img_path)[:,:,0:3]
    score = WhiteContain(img)
    if score < thresh:
        valid(img_path)
    else:
        discard(img_path)

CheckOrCreate('./discard')
CheckOrCreate('./valid')

files = glob('*.png')

map(OpenFileCheckMove, files)
