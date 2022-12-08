import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

#initializing predictor matrix for simple, linear, formula-fitted GCAMP expression
def predic_mat_init(output_size):
    predic_mat  = [[0 for x in range(3)] for i in range(output_size[1])]
    return predic_mat

#initializes matrix for the binary offset, right now it only contains 0's
def predic_mat_binary_init(stimulus, output_size, lag_limit, withheld_stim):
    for i in range(np.shape(stimulus)[0]):
        remove_lag = 0
        if stimulus[i] == withheld_stim:
            remove_lag = lag_limit[i][1] - lag_limit[i][0]
    total_sum = 0
    for i in range(np.shape(lag_limit)[0]):
        col_sum = lag_limit[i][1] - lag_limit[i][0]
        total_sum += col_sum
    
    if type(output_size) == int:
        predic_mat  = [[0 for x in range(total_sum - remove_lag)] for i in range(output_size)]
    else:
        predic_mat  = [[0 for x in range(total_sum - remove_lag)] for i in range(output_size[1])]
    return predic_mat

#finding the minimum value for fluorscence so that we can somewhat adjust our predicted output
def find_min(output_data, output_size, cell_number):
    min = 10000
    for i in range(output_size[1]): 
        if output_data[cell_number][i] < min:
            min = output_data[cell_number][i]
    return min

#Gcamp decay function
def Gcamp_decay(time, tau):
    output = round(((1/tau) * np.exp(-time/tau)), 3)
    return output

# function for creating a moving average vector of a noisy vector
def moving_average(data, desired_interval):
    moving_average = np.convolve(data, np.ones(desired_interval), 'same') / desired_interval
    return moving_average

# function that applies moving average calculations but for an entire matrix, performs the operations on each row
def moving_average_mat(data, desired_interval):
    for i in range(np.shape(data)[0]):
        data[i] = moving_average(data[i], desired_interval)
    return data

# Creates the final predictor matrix which we will run through the regression. This function creates the
# predictor matrix for the more simple, formula-fitted GCAMP expression 
def predic_mat_gen(read_data, size, predic_mat, cutoff, time_interval, frequency):
    for i in range(size[0]):
        predic_mat_size = np.shape(predic_mat)
        if read_data[i][0] == "plus":
            pos_time = read_data[i][1]
            pos_samp = int(pos_time * frequency)
            for j in range(cutoff):
                if (j+pos_samp) < predic_mat_size[0]:
                    predic_mat[pos_samp+j][0] = Gcamp_decay(j * time_interval, 2)
        elif read_data[i][0] == "minus":
            neg_time = read_data[i][1]
            neg_samp = int(neg_time * frequency)
            for j in range(cutoff):
                if (j+neg_samp) < predic_mat_size[0]:
                    predic_mat[neg_samp+j][1] = Gcamp_decay(j * time_interval, 2)
        elif read_data[i][0] == "licks":
            lick_time = read_data[i][1]
            lick_samp = int(lick_time * frequency)
            for j in range(cutoff):
                if (j+lick_samp) < predic_mat_size[0]:
                    predic_mat[lick_samp+j][2] = Gcamp_decay(j * time_interval, 2) 
    return predic_mat

# Creates the final predictor matrix with binary offset. When a stimulus occurs, a one will be placed at that time. 
# Additionally, a diagonal of one's will be created for however many columns we determined in the lag_limit. 
# This will be used for our reduced rank regression 

#Reduced Rank Regression solution (not the most computationally efficient approach but it is the direct mathematical derivation))
#Not sure if we want to apply the regression to the normalized, moving average data, or the raw output. Same consideration when plotting.
#RRR is a multivariable regression performed on the output data from all of the cells and generates coefficients for each cell. 
# RRR artificially restricts the rank of by only using the first r columns of the singular-value decomposed matrix of the ordinary solution
# done to reduce overfitting and computational complexity  
def rrr_formula(regress_mat_binary, output_data, r):
    regress_mat_binary_transpose = np.transpose(regress_mat_binary)
    reg_tp_reg_inv = np.linalg.inv(np.dot(regress_mat_binary_transpose, regress_mat_binary))
    ordinary_lin_reg = np.dot(reg_tp_reg_inv, np.transpose(regress_mat_binary)) 
    ordinary_lin_reg = np.dot(ordinary_lin_reg, np.transpose(output_data)) # the calculations up to this point have been for the OLS which we will modify to get the RRR
    U, D, VT = np.linalg.svd(np.dot(regress_mat_binary, ordinary_lin_reg)) #Singular Value Decomposition 
    VT_r = VT[:(r), :]
    rrr_sol = np.dot(ordinary_lin_reg, np.transpose(VT_r))
    rrr_sol = np.dot(rrr_sol, VT_r)
    return rrr_sol

def predic_mat_gen_binary(read_data, size, predic_mat_binary, frequency, stimuli, lag_limit, withheld_stim, lambda_stim, lambda_coeff):
    lag_sum = 0
    del_lag_limit = 0
    for i in range(np.shape(stimuli)[0]):
        del_lag_limit = (lag_limit[i][1] - lag_limit[i][0])
        for j in range(size[0]):
            predic_mat_size = np.shape(predic_mat_binary)
            if stimuli[i] == withheld_stim:
                stimuli[i] = None
            elif stimuli[i] == lambda_stim:
                stimuli[i] == None
            if read_data[j][0] == stimuli[i]:
                stim_time = read_data[j][1]
                stim_samp = int(stim_time * frequency)
                for k in range(del_lag_limit):
                    if (k+stim_samp) < predic_mat_size[0]:
                        predic_mat_binary[stim_samp+k][lag_sum + k] = 1
            elif read_data[j][0] == lambda_stim:
                stim_time = read_data[j][1]
                stim_samp = int(stim_time * frequency)
                for k in range(del_lag_limit):
                    if (k+stim_samp) < predic_mat_size[0]:
                        predic_mat_binary[stim_samp+k][lag_sum + k] = lambda_coeff
        lag_sum = lag_sum + del_lag_limit
    return predic_mat_binary

def predic_mat_gen_binary_dic(read_data, predic_mat_binary, frequency, stimuli, lag_limit, withheld_stim, lambda_stim, lambda_coeff):
    lag_sum = 0
    del_lag_limit = 0
    for i in range(np.shape(stimuli)[0]):
        modify = False
        del_lag_limit = (lag_limit[i][1] - lag_limit[i][0])
        predic_mat_size = np.shape(predic_mat_binary)
        if stimuli[i] == withheld_stim:
            stimuli[i] = None
        elif stimuli[i] == lambda_stim:
            modify = True
        if stimuli[i] in read_data.keys():
            stim_samp = read_data[stimuli[i]] * frequency # stim samp is a 2d array with each entry a row
            for j in range(np.shape(read_data[stimuli[i]])[0]):
                stim_samp_element = int(stim_samp[j][0])
                if modify == True:
                    for k in range(del_lag_limit):
                        if (k+stim_samp_element) < predic_mat_size[0]:
                            predic_mat_binary[stim_samp_element+k][lag_sum + k] = lambda_coeff
                else:
                    for k in range(del_lag_limit):
                        if (k+stim_samp_element) < predic_mat_size[0]:
                            predic_mat_binary[stim_samp_element+k][lag_sum + k] = 1
        lag_sum = lag_sum + del_lag_limit
    return predic_mat_binary
