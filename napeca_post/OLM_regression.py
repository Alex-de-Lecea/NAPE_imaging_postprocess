import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import predictor_matrix_gen
import scipy.io as sio

from plotly.subplots import make_subplots
import plotly.graph_objects as go

#User variables
practice_inputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/event_times_VJ_OFCVTA_7_260_D6_trained.csv')
practice_outputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/VJ_OFCVTA_7_260_D6_neuropil_corrected_signals_15_50_beta_0.8.csv')

# practice_inputs = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\pl01o1d1events')
# practice_outputs = sio.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\Fall')


frequency = 15
time_interval = round(1/(frequency), 1)
mov_avg_int = 20 #the size of the interval we wish to use for our moving average calculations
cell_number = 3 #The cell we wish to see when plotting. A cell that is known to be more correlated to activity will have more accurate predicted curve
reduced_rank = 20 #the limit of ranks we wish to choose when performing our rrr. Lower ranks remove more noise but can show less of the actual data as well

stimuli = ['plus', 'minus', 'licks']
lag_limit= [[0, 5], [0, 5], [0, 5]]

# stimuli = ['cue', 'cue1', 'cue2', 'cue2r', 'cue2u', 'cue3' ,'reward']
# lag_limit = [[0, 75], [0, 75], [0, 75], [0, 75], [0, 75], [0, 75], [0, 75]]

withheld_stim = None
lambda_stim = None
lambda_coeff = 0.5

# Load data
if (type(practice_inputs) and type(practice_outputs)) == dict:
    #data inputs
    read_data = practice_inputs
    output_data = practice_outputs['F']

    #calculated variables
    output_size = np.shape(output_data)[1]

    #predictor matrix initialization and generation
    predic_mat_binary = predictor_matrix_gen.predic_mat_binary_init(stimuli, output_size, lag_limit, withheld_stim)
    regress_mat_binary = predictor_matrix_gen.predic_mat_gen_binary_dic(read_data, predic_mat_binary, frequency, stimuli, lag_limit, withheld_stim)

else:
    #data inputs 
    read_data = pd.read_csv(practice_inputs)
    read_data = pd.DataFrame.to_numpy(read_data)

    output_data = pd.read_csv(practice_outputs)
    output_data = pd.DataFrame.to_numpy(output_data)

    #calculated variables
    size = np.shape(read_data)
    output_size = np.shape(output_data)

    #predictor matrix initialization and generation
    predic_mat_binary = predictor_matrix_gen.predic_mat_binary_init(stimuli, output_size, lag_limit, withheld_stim)
    regress_mat_binary = predictor_matrix_gen.predic_mat_gen_binary(read_data, size, predic_mat_binary, frequency, stimuli, lag_limit, withheld_stim, lambda_stim, lambda_coeff)

#moving average calculations
moving_average_output = predictor_matrix_gen.moving_average(output_data[cell_number], mov_avg_int) #moving average of the output of one cell (this is fine for testing but we are going to eventually use the matrix of all of the cell activity)
moving_average_output_total = predictor_matrix_gen.moving_average_mat(output_data, mov_avg_int)
min = np.amin(moving_average_output)

#1-D Linear regression (takes the output of one cell and fits our predictor matrix to it)
olsmod = sm.OLS((moving_average_output - min), regress_mat_binary)
olsres = olsmod.fit()
ypred = olsres.predict(regress_mat_binary)

#Reduced Rank Regression Calculations 
rrr_sol = predictor_matrix_gen.rrr_formula(regress_mat_binary, moving_average_output_total, reduced_rank)
rrr_ypred = np.dot(predic_mat_binary, np.transpose(rrr_sol)[cell_number])

#plotting predictor matrix
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
data2D = regress_mat_binary
im = plt.imshow(data2D, cmap="copper_r")
plt.colorbar(im)
plt.show()

predic_mat_size = np.shape(predic_mat_binary)

#plot creation with plotly (we can only plot one given cell and its predicted output at a time)
# x = np.array(range(0, predic_mat_size[0]))

# fig = make_subplots(rows=2, cols=1,
#                     subplot_titles = ("Moving Average of Actual Fluoresence", "Predicted Fluoresence"),
#                     shared_yaxes=True,
#                     horizontal_spacing=0.02)

# fig['layout']['xaxis2']['title'] = "Sample number"
# fig['layout']['yaxis']['title'] = "Fluoresence"
# fig.add_trace(go.Scattergl(x=x, y=(moving_average_output_total[cell_number]), mode='lines'), row=1, col=1) # here is where we choose whether to use moving average or raw
# fig.add_trace(go.Scattergl(x=x, y=np.transpose(rrr_ypred), mode='lines'), row=2, col=1) #we took the tranpose of rrr_ypred for plotting purposes

# fig.update_layout(height=600, width=1200)
# fig.update_xaxes(matches='x')

# #showing the output
# fig.show()
print(olsres.summary())
print(rrr_sol) #Each column corresponds to the coefficients of the given cell
print(np.corrcoef(rrr_ypred, moving_average_output_total[cell_number]))
print(np.corrcoef(ypred, moving_average_output_total[cell_number]))
