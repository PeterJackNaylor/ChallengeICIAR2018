import sys
from skimage.io import imread
from WrittingTiff.Extractors import bin_analyser
from WrittingTiff.Extractors import PixelSize, MeanIntensity, Centroid


list_f = [PixelSize("Pixel_sum", 0), MeanIntensity("Intensity_mean_0", 0), 
          MeanIntensity("Intensity_mean_5", 5), Centroid(["Centroid_x", "Centroid_y"], 0)]

path = sys.argv[1]
rgb = imread(path)
pred = imread(sys.argv[2])

table_feat = bin_analyser(rgb, pred, list_f, pandas_table=True)
table_feat.to_csv(path.replace('.png', '.csv'))