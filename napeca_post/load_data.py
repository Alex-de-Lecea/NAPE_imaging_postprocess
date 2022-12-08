import scipy.io as sio
import numpy as np
import pandas as pd
import os

mat = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\Fall')
lab = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\pl01o1d1events')
maps = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\pl01o1d1maps')
fdir = os.path.abspath('./sample_data/PL01/o1d1')

# print(type(mat))
print(mat.keys())
print(lab.keys())
print(np.shape(lab['lick']))
# print(lab['cue2u'])
# print(np.shape(mat['F']))
# print(mat['ops']['fs'])

# practice_inputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/event_times_VJ_OFCVTA_7_260_D6_trained.csv')
# sio.savemat('c:/Users/Alex_dl/Documents/GitHub/NAPE_imaging_postprocess/napeca_post/sample_data/PL01/o1d1/2d_data.mat', mdict={'arr': practice_inputs})