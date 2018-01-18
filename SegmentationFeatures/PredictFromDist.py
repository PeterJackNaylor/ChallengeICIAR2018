from Nets.UNetDistance import UNetDistance
import numpy as np
from UsefulFunctions.ImageTransf import Transf
from skimage.io import imread
from Deprocessing.Morphology import PostProcess
from UsefulFunctions.RandomUtils import add_contours, color_bin
from os.path import join
from skimage.io import imsave
from glob import glob
from os.path import basename
import sys
import os
img_path = sys.argv[1]
WEIGHT_LOG = sys.argv[2]
MEAN_FILE = sys.argv[3]

def UNetAugment(img):
    Tr = Transf('')
    out = Tr.enlarge(img, 92, 92)
    return out

def PP(dist, p1, p2, rgb=None, save_path=None, name=None):
    PRED = PostProcess(dist, p1, p2)
    if rgb is not None:
        prob_n = join(save_path, "dist_{}.png").format(name)
        pred_n = join(save_path, "pred_{}.png").format(name)
        c_pr_n = join(save_path, "C_pr_{}.png").format(name)
        ## CHECK PLOT FOR PROB AS IT MIGHT BE ILL ADAPTED

        imsave(prob_n, dist)
        imsave(pred_n, color_bin(PRED))
        imsave(c_pr_n, add_contours(rgb, PRED))
    return PRED



img = imread(img_path).astype('uint8')
size = img.shape[:2]
model = UNetDistance("", 
                     BATCH_SIZE=1,
                     IMAGE_SIZE=size,
                     NUM_CHANNELS=3,
                     LOG=WEIGHT_LOG,
                     N_FEATURES=16)



img_a = UNetAugment(img)[0:-12,0:-12]
img_f = img_a.astype(float) - np.load(MEAN_FILE)
Xval = img_f[np.newaxis, :]





feed_dict = {model.input_node: Xval,
             model.is_training: False}
pred = model.sess.run([model.predictions],
                       feed_dict=feed_dict)
pred = pred[0][0]
pred[pred < 0] = 0
pred = pred.astype('uint8')
os.mkdir("out")
out = PP(pred, 1, 0, rgb=img[0:-12,0:-12], save_path="out", name=basename(img_path).replace('.tif', ''))
