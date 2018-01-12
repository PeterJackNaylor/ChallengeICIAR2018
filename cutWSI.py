import openslide
from openslide import open_slide # http://openslide.org/api/python/
from skimage.transform import resize
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imread
from TissueSegmentation import ROI_binary_mask

img_path = ''
xml_path = ''
GT_path = ''
number_of_imgs = 

scan = openslide.OpenSlide(img_path)
last_dim_n = len(scan.level_dimensions)
last_dim = scan.level_dimensions[last_dim_n]

whole_img = scan.read_region((0,0), last_dim_n, last_dim)
whole_img = np.array(whole_img)[:,:,0:3]


small_img = resize(whole_img, (500, 500))
small_img = img_as_ubyte(small_img)
mask = ROI_binary_mask(small_img, ticket=(80,80,80))
mask_tissue = resize(mask, last_dim)
mask_tissue = img_as_ubyte(mask_tissue)
mask_tissue[mask_tissue > 0] = 255

GT_file = imread(GT_path)
GT_file[GT_file > 0] = 255
Red_area = GT_file[:,:,0]
Green_area = GT_file[:,:,1]
Blue_area = GT_file[:,:,2]

mask_tissue = mask_tissue - Red_area - Green_area - Blue_area

channels = (mask_tissue, Red_area, Green_area, Blue_area)
for c in channels:
    # for each channel, pick a cube, insert in img, remove from old image, continue







