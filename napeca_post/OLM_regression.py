import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import predictor_matrix_gen

from plotly.subplots import make_subplots
import plotly.graph_objects as go

#User variables
practice_inputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/event_times_VJ_OFCVTA_7_260_D6_trained.csv')
practice_outputs = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/VJ_OFCVTA_7_260_D6_neuropil_corrected_signals_15_50_beta_0.8.csv')
frequency = 5
mov_avg_int = 20 #the size of the interval we wish to use for our moving average calculations
cutoff = 20 #number of seconds after stimulus we wish to see the fluoresence. Higher cutoff will produce less error. 
cell_number = 3 #The cell we wish to see when plotting. A cell that is known to be more correlated to activity will have more accurate predicted curve
lag_limit = 25 #number of samples after stimulus we wish to see the fluoresence for the binary regression matrix. Higher lag_limit produces much better results

#Load data
read_data = pd.read_csv(practice_inputs)
read_data = pd.DataFrame.to_numpy(read_data)

output_data = pd.read_csv(practice_outputs)
output_data = pd.DataFrame.to_numpy(output_data)

#Calculated variables
size = np.shape(read_data)
time_interval = round(1/(frequency), 1)
output_size = np.shape(output_data)
cutoff = cutoff * frequency #number of samples we wish to continue seeing the gcamp after the initial stimulus (cutoff of 10 samples at 5hz is 2 seconds)

#Initialize predictor matrix
#This first predic_mat and regress_mat use the first method we implemented where we use a predetermined formula for GCAMP expression and fit the entire
#curve to the data linearly. This method produces less over-fitting and is much faster computationally but is less accurate since it assumes consistent, formulaic
#GCAMP traces
predic_mat = predictor_matrix_gen.predic_mat_init(output_size)
regress_mat = predictor_matrix_gen.predic_mat_gen(read_data, size, predic_mat, cutoff, time_interval, frequency)

predic_mat_binary = predictor_matrix_gen.predic_mat_binary_init(output_size, lag_limit)
regress_mat_binary = predictor_matrix_gen.predic_mat_gen_binary(read_data, size, predic_mat_binary, frequency, lag_limit)

predic_mat_size = np.shape(predic_mat)

#Moving average calculations
moving_average_output = predictor_matrix_gen.moving_average(output_data[cell_number], mov_avg_int) #moving average of the output of one cell (this is fine for testing but we are going to eventually use the matrix of all of the cell activity)
moving_average_output_total = predictor_matrix_gen.moving_average_mat(output_data, mov_avg_int)
min = np.amin(moving_average_output)

#1-D Linear regression (takes the output of one cell and fits our predictor matrix to it)
olsmod = sm.OLS((moving_average_output - min), regress_mat_binary)
olsres = olsmod.fit()
ypred = olsres.predict(regress_mat_binary)

#Reduced Rank Regression solution (not the most computationally efficient approach but it is the direct mathematical derivation))
#Not sure if we want to apply the regression to the normalized, moving average data, or the raw output. Same consideration when plotting.
#RRR is a multivariable regression performed on the output data from all of the cells and generates coefficients for each cell. This is different to the
#1-D linear regression where we simply fit the output of one cell
regress_mat_binary_transpose = np.transpose(regress_mat_binary)
reg_tp_reg_inv = np.linalg.inv(np.dot(regress_mat_binary_transpose, regress_mat_binary))
ordinary_lin_reg = np.dot(reg_tp_reg_inv, np.transpose(regress_mat_binary)) 
ordinary_lin_reg = np.dot(ordinary_lin_reg, np.transpose(moving_average_output_total)) # the calculations up to this point have been for the OLS which we will modify to get the RRR
U, D, VT = np.linalg.svd(np.dot(regress_mat_binary,ordinary_lin_reg)) #Singular Value Decomposition 
rrr_sol = np.dot(ordinary_lin_reg,np.transpose(VT))
rrr_sol = np.dot(rrr_sol, VT) #final solution, matrix containing the coefficients for each cell

#plot creation with plotly (we can only plot one given cell and its predicted output at a time)
x = np.array(range(0, predic_mat_size[0]))
pos_stim_arr = np.transpose(regress_mat_binary)[0]
neg_stim_arr = np.transpose(regress_mat_binary)[lag_limit]
lick_stim_arr = np.transpose(regress_mat_binary)[2*lag_limit]
rrr_ypred = np.dot(predic_mat_binary, np.transpose(rrr_sol)[cell_number])

fig = make_subplots(rows=5, cols=1,
                    subplot_titles = ("CS+ Predicted", "CS- Predicted", "Licks Predicted", "Moving Average of Actual Fluoresence", "Predicted Fluoresence"),
                    shared_yaxes=True,
                    horizontal_spacing=0.02)

fig['layout']['xaxis5']['title'] = "Sample number"
fig['layout']['yaxis3']['title'] = "Fluoresence"
fig.add_trace(go.Scattergl(x=x, y=(pos_stim_arr), mode='lines'), row=1, col=1)
fig.add_trace(go.Scattergl(x=x, y=(neg_stim_arr), mode='lines'), row=2, col=1)
fig.add_trace(go.Scattergl(x=x, y=(lick_stim_arr), mode='lines'), row=3, col=1)
fig.add_trace(go.Scattergl(x=x, y=(moving_average_output_total[cell_number]), mode='lines'), row=4, col=1) # here is where we choose whether to use moving average or raw
fig.add_trace(go.Scattergl(x=x, y=np.transpose(rrr_ypred), mode='lines'), row=5, col=1) #we took the tranpose of rrr_ypred for plotting purposes

fig.update_layout(height=600, width=1200)
fig.update_xaxes(matches='x')

#showing the output
fig.show()
print(olsres.summary())
print(rrr_sol) #Each column corresponds to the coefficients of the given cell
