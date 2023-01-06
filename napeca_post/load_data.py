import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import predictor_matrix_gen
import scipy.io as sio

from plotly.subplots import make_subplots
import plotly.graph_objects as go

cell_number = 3

mat = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\Fall')
lab = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\pl01o1d1events')
maps = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\pl01o1d1maps')
fdir = os.path.abspath('./sample_data/PL01/o1d1')

# print(type(mat))
print(mat.keys())
print(lab.keys())
print(maps.keys())
# print(lab['cue2u'])
# print(np.shape(mat['F']))
# print(mat['ops']['fs'])

practice_inputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/event_times_VJ_OFCVTA_7_260_D6_trained.csv')
read_data = pd.read_csv(practice_inputs)
read_data = pd.DataFrame.to_numpy(read_data)
# sio.savemat('c:/Users/Alex_dl/Documents/GitHub/NAPE_imaging_postprocess/napeca_post/sample_data/PL01/o1d1/2d_data.mat', mdict={'arr': practice_inputs})

practice_outputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/VJ_OFCVTA_7_260_D6_neuropil_corrected_signals_15_50_beta_0.8.csv')
zhou_data = pd.read_csv(practice_outputs)

#full data
zhou_data = pd.DataFrame.to_numpy(zhou_data)
matlab_data = mat['F']

#specific trial zhou_data
reward_samp_zhou = int(read_data[cell_number][1])*5
zhou_data_reward1 = zhou_data[:, reward_samp_zhou:(reward_samp_zhou + 25)]

#specific trial matlab_data
reward_samp_matlab = int(lab['reward'][cell_number])*15
matlab_data_reward1 = matlab_data[:, reward_samp_matlab:(reward_samp_matlab + 75)]

#plotting raw data matrix
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
data2D = matlab_data
im = plt.imshow(data2D, cmap="gnuplot")
plt.colorbar(im)
plt.show()
