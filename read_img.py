from pandas import read_csv

PATH = "../Photos"
CSV = "../Photos/microscopy_ground_truth.csv"


def Map(classs):
    if classs == 'Normal':
        return 0
    elif classs == 'Benign':
        return 1
    elif classs == 'InSitu':
        return 2
    elif classs == 'Invasive':
        return 3
    else:
        return -1

def ArgMap(val):
    if val == 0:
        return 'Normal'
    elif val == 1:
        return 'Benign'
    elif val == 2:
        return 'InSitu'
    elif val == 3:
        return 'Invasive'
    else:
        return 'NonAssigned'

def TableMetaData(path=PATH, csv=CSV):
    table = read_csv(csv, header=None)
    table.columns = ['name', 'class']
    table['y'] = table.apply(lambda row: Map(row['class']), axis=1)
    return table

MetaTable = TableMetaData(PATH, CSV)