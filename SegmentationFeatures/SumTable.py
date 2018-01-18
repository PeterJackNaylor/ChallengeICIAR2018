import sys
from os.path import basename
from pandas import read_csv, concat
import pdb

input_name = sys.argv[1]
input_table = read_csv(input_name, index_col=0)
Centers = input_table[["Centroid_x", "Centroid_y"]]
input_table = input_table.drop(["Centroid_x", "Centroid_y"], axis=1)

## Add more descripter if needed
desc = input_table.describe()
desc = desc.drop('count', axis=0)

n = desc.shape[0]
series_to_concat = []
for i in range(n):
    s = desc.ix[i]
    s = s.rename(lambda ind: s.name + "_" + ind)
    series_to_concat.append(s)
vec = concat(series_to_concat, axis=0)

lbl = ["Normal", "Benign", "Invasive", "InSitu"]
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in lbl])))
label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

tags = input_name
if basename(tags)[0] == "n":
    pre = "Normal"
elif basename(tags)[0] == "b":
    pre = "Benign"
elif basename(tags)[0:2] == "iv":
    pre = "Invasive"
elif basename(tags)[0:2] == "is":
    pre = "InSitu"


vec['label'] = label_map[pre]
output_name = sys.argv[2]
vec.to_csv(output_name)
