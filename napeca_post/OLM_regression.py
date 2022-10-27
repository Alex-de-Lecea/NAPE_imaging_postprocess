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
cutoff = 8 #number of seconds after stimulus we wish to see the fluoresence. Higher cutoff will produce less error. 
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

min = predictor_matrix_gen.find_min(output_data, output_size, cell_number)
moving_average_output = predictor_matrix_gen.moving_average(output_data[cell_number], 3)
moving_average_pos = predictor_matrix_gen.moving_average(regress_mat[0], 10)
moving_average_pos = predictor_matrix_gen.moving_average(regress_mat[1], 10)

#Linear regression 
regress_mat = regress_mat 
output_data[cell_number] = output_data[cell_number]
olsmod = sm.OLS(output_data[cell_number], regress_mat)
olsres = olsmod.fit()
ypred = olsres.predict(regress_mat)

print(olsres.summary())
