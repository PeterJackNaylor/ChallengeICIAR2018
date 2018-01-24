import matplotlib
matplotlib.use('agg')
import openslide
from openslide import open_slide # http://openslide.org/api/python/
from skimage.transform import resize
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from TissueSegmentation import ROI_binary_mask
import UsefulFunctions.UsefulOpenSlide as UOS
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from os.path import basename
import sys
import os
import pdb

def visualise_cut(slide, list_pos, res_to_view=None, color='red', size=12, title=""):
    map_col = {0: "white", 1:"red", 2:"green", 3:"blue"}
    if res_to_view is None:
        res_to_view = slide.level_count - 3
    whole_slide = np.array(slide.read_region(
        (0, 0), res_to_view, slide.level_dimensions[res_to_view]))
    max_x, max_y = slide.level_dimensions[res_to_view]
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, aspect='equal')
    # ax.imshow(flip_vertical(whole_slide))  # , origin='lower')
    # whole_slide = flip_horizontal(whole_slide)
    ax.imshow(whole_slide)
    for para in list_pos:
        top_left_x, top_left_y = UOS.get_X_Y_from_0(
            slide, para[0], para[1], res_to_view)
        w, h = UOS.get_size(slide, para[2], para[3], para[4], res_to_view)
        p = patches.Rectangle(
            (top_left_x, max_y - top_left_y - h), w, h, fill=False, edgecolor=map_col[para[5]])
        p = patches.Rectangle((top_left_x, top_left_y), w,
                              h, fill=False, edgecolor=map_col[para[5]])
        ax.add_patch(p)
    ax.set_title(title, size=20)
    return fig


dic = {0:0, 1:0, 2:0, 3:0} # counts

fold = sys.argv[1]
num = int(sys.argv[2])
number_of_imgs = int(sys.argv[3])
os.mkdir('./patch_extract/')
os.mkdir('./samples/')

img_path = os.path.join(fold, 'A{:02d}.svs').format(num)


out_path = './patch_extract/' + basename(img_path).replace('.svs', '.png')
GT_path = os.path.join(fold, 'gt_thumbnails/' + basename(img_path).replace('.svs', '.png'))


scan = openslide.OpenSlide(img_path)
last_dim_n = len(scan.level_dimensions) - 1
last_dim = scan.level_dimensions[last_dim_n]
last_dim_inv = (last_dim[1], last_dim[0])
whole_img = scan.read_region((0,0), last_dim_n, last_dim)
whole_img = np.array(whole_img)[:,:,0:3]


small_img = resize(whole_img, (500, 500))
small_img = img_as_ubyte(small_img)
mask = ROI_binary_mask(small_img, ticket=(80,80,80))
mask = mask.astype('uint8')
mask[mask > 0] = 255
mask_tissue = resize(mask, last_dim_inv, order=0)
mask_tissue = img_as_ubyte(mask_tissue)
mask_tissue[mask_tissue > 0] = 255

GT_file = imread(GT_path)
GT_file[GT_file > 0] = 255
Red_area = GT_file[:,:,0]
Red_area = img_as_ubyte(resize(Red_area, last_dim_inv, order=0))
Green_area = GT_file[:,:,1]
Green_area = img_as_ubyte(resize(Green_area, last_dim_inv, order=0))
Blue_area = GT_file[:,:,2]
Blue_area = img_as_ubyte(resize(Blue_area, last_dim_inv, order=0))

mask_tissue = mask_tissue - Red_area - Green_area - Blue_area
s_0_x, s_0_y = UOS.get_size(scan, 224, 224, 0, last_dim_n)

channels = (mask_tissue, Red_area, Green_area, Blue_area)

list_channel = []
for k, c in enumerate(channels):
    i = 0
    c_cop = c.copy()
    while ((c_cop > 0).sum() > 0 and i < number_of_imgs):
        x_l, y_l = np.where(c > 0)  
        pic = random.randint(0, len(x_l))
        x = x_l[pic] - s_0_x // 2
        y = y_l[pic] - s_0_y // 2
        c_cop[(x-s_0_x):(x+s_0_x), (y-s_0_y):(y+s_0_y)] = 0
        x_0, y_0 = UOS.get_X_Y(scan, x, y, last_dim_n)
        # add cube to final list: label, x,y for read_imd
        list_channel.append([y_0, x_0, 224, 224, 0, k])
        i += 1


for y_0, x_0, s_x, s_y, l, lbl in list_channel:
    coord_0 = UOS.get_X_Y(scan, y_0, x_0, last_dim_n)
    img = scan.read_region((y_0, x_0), l, (s_x, s_y))
    imsave('samples/{}_{}_{}.png'.format(lbl, num, dic[lbl]), img)
    dic[lbl] += 1
fif = visualise_cut(scan, list_channel, res_to_view=last_dim_n)
fif.savefig(out_path, bbox_inches='tight')
    #    for each channel, pick a cube, insert in img, remove from old image, continue







