import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import predictor_matrix_gen

#user variables
practice_inputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/event_times_VJ_OFCVTA_7_260_D6_trained.csv')
practice_outputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/VJ_OFCVTA_7_260_D6_neuropil_corrected_signals_15_50_beta_0.8.csv')
frequency = 5
cutoff = 2 #number of seconds after stimulus we wish to see the fluoresence. Higher cutoff might produce less error. 
cell_number = 3

#load data
read_data = pd.read_csv(practice_inputs)
read_data = pd.DataFrame.to_numpy(read_data)

output_data = pd.read_csv(practice_outputs)
output_data = pd.DataFrame.to_numpy(output_data)

#calculated variables
size = np.shape(read_data)
time_interval = round(1/(frequency), 1)
output_size = np.shape(output_data)
cutoff = cutoff * frequency #number of samples we wish to continue seeing the gcamp after the initial stimulus (cutoff of 10 samples at 5hz is 2 seconds)

#initialize predictor matrix
predic_mat = predictor_matrix_gen.predic_mat_init(output_size)
regress_mat = predictor_matrix_gen.predic_mat_gen(read_data, size, predic_mat, cutoff, time_interval)

predic_mat_size = np.shape(predic_mat)

#Linear regression 
olsmod = sm.OLS(output_data[cell_number], regress_mat)
olsres = olsmod.fit()
ypred = olsres.predict(regress_mat)

pos_stim_arr = np.transpose(regress_mat)[0]
neg_stim_arr = np.transpose(regress_mat)[0]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

#plotting pos_stim_arr over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("CS+ expected fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
ax1.plot(x, pos_stim_arr, color = "red", marker = "o")
plt.legend()
plt.show()

#plotting neg_stim_arr over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("CS- expected fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
ax2.plot(x, neg_stim_arr, color = "red", marker = "o")
plt.legend()
plt.show()

#plotting actual fluoresence over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("Actual fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
ax3.plot(x, output_data[3], color = "red", marker = "o")
plt.legend()
plt.show()

#plotting expected fluoresence over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("Expected fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
ax4.plot(x, ypred, color = "red", marker = "o")
plt.legend()
plt.show()
