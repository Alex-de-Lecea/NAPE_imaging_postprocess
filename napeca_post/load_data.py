import scipy.io as sio
import numpy as np

mat = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\Fall')
lab = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\pl01o1d1events')
licks = np.load(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\2022-02-04\licks.times.npy')

print(type(mat))
print(mat.keys())
print(np.shape(mat['F']))
print(lab.keys())
print(np.shape(lab['lick']))
print(lab['lick'])
