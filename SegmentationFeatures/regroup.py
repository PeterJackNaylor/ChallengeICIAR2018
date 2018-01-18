import sys
from glob import glob
import pandas as pd


files = glob('*.csv')
all_tab = []
for name in files:
    tbl = pd.read_csv(name, header=None, index_col=0)
    tbl.columns = [name.split('_')[1].split('.')[0]]
    all_tab.append(tbl)
aggr = pd.concat(all_tab, axis=1).T
output = sys.argv[1]
aggr.to_csv(output)

