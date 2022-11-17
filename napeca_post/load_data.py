import scipy.io as sio
import numpy as np

mat = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\Fall')

print(type(mat))
# print(mat.items())
print(mat['redcell'])