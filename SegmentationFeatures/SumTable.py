import sys
from pandas import read_csv


input_name = sys.argv[1]
input_table = read_csv(input_name, index_col=0)
output_name = sys.argv[2]
