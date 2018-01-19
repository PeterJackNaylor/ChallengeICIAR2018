import sys
import numpy as np
from scipy import stats
table_name = sys.argv[1]
TYPE = sys.argv[2]
out_name = sys.argv[3]

table_npy = np.load(table_name)
id = table_npy[:,0].copy()
y = table_npy[:,1].copy()
X = table_npy[:,2:]

n, p = X.shape
step = 2048
K = p // 2048
if TYPE == "Max":
    new_res = np.zeros((n, 2048 + 2), dtype='float')
    new_res[:,0] = id
    new_res[:,1] = y
    for j in range(2048):
        new_res[:,2+j] = np.max(X[:,K*j:K*(j+1)], axis=1)
    np.save(out_name, new_res)
elif TYPE == "Mean":
    new_res = np.zeros((n, 2048 + 2), dtype='float')
    new_res[:,0] = id
    new_res[:,1] = y
    for j in range(2048):
        new_res[:,2+j] = np.mean(X[:,K*j:K*(j+1)], axis=1)
    np.save(out_name, new_res)
elif TYPE == "Median":
    new_res = np.zeros((n, 2048 + 2), dtype='float')
    new_res[:,0] = id
    new_res[:,1] = y
    for j in range(2048):
        new_res[:,2+j] = np.median(X[:,K*j:K*(j+1)], axis=1)
    np.save(out_name, new_res)
elif TYPE == "All":
    new_res = np.zeros((n, 2048 + 2), dtype='float')
    new_res[:,0] = id
    new_res[:,1] = y
    for j in range(2048):
        s_description = stats.describe(X[:,K*j:K*(j+1)], axis=1)
    np.save(out_name, new_res)