import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import predictor_matrix_gen
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#user variables
practice_inputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/event_times_VJ_OFCVTA_7_260_D6_trained.csv')
practice_outputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/VJ_OFCVTA_7_260_D6_neuropil_corrected_signals_15_50_beta_0.8.csv')
frequency = 5
cutoff = 20 #number of seconds after stimulus we wish to see the fluoresence. Higher cutoff will produce less error. 
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
regress_mat = predictor_matrix_gen.predic_mat_gen(read_data, size, predic_mat, cutoff, time_interval, frequency)

predic_mat_size = np.shape(predic_mat)

#Linear regression 
regress_mat = regress_mat #adding the min is supposed to bring our predicted values up by whatever the minimum value is so we ignore that background noise
output_data[cell_number] = output_data[cell_number]
olsmod = sm.OLS(output_data[cell_number], regress_mat)
olsres = olsmod.fit()
ypred = olsres.predict(regress_mat)

# moving averages
min = predictor_matrix_gen.find_min(output_data, output_size, cell_number)
moving_average_output = predictor_matrix_gen.moving_average(output_data[cell_number], 20)
moving_average_pos = predictor_matrix_gen.moving_average(regress_mat[0], 10)
moving_average_pos = predictor_matrix_gen.moving_average(regress_mat[1], 10)
moving_average_ypred = predictor_matrix_gen.moving_average(ypred, 10)


pos_stim_arr = np.transpose(regress_mat)[0]
neg_stim_arr = np.transpose(regress_mat)[1]

#array for time (samples) which we use for our x-axis
x = np.array(range(0, predic_mat_size[0]))

# fig = make_subplots(rows=1, cols=2,
#                     shared_yaxes=True,
#                     horizontal_spacing=0.02)

# fig.add_trace(go.Scattergl(x=x, y=output_data[cell_number], mode='markers'), row=1, col=1)
# fig.add_trace(go.Scattergl(x=x, y=ypred, mode='markers'), row=1, col=2)

# fig.update_layout(height=600, width=1200)
# fig.update_xaxes(matches='x')

# fig.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
plt.subplot(2,2,1)
plt.xticks(np.arange(0, len(x) + 1)[::5000], x[::5000])
plt.plot(x, pos_stim_arr)
plt.title('CS+ Expected Fluoresence')

plt.subplot(2,2,3)
plt.xticks(np.arange(0, len(x) + 1)[::5000], x[::5000])
plt.plot(x, neg_stim_arr)
plt.title('CS- Expected Fluoresence')

plt.subplot(2,2,2)
plt.xticks(np.arange(0, len(x) + 1)[::5000], x[::5000])
plt.plot(x, (moving_average_output - min))
plt.title('Actual Fluoresence')

plt.subplot(2,2,4)
plt.xticks(np.arange(0, len(x) + 1)[::5000], x[::5000])
plt.plot(x, ypred)
plt.title('Predicted Fluoresence')

plt.show()