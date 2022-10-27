import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

#initializing predictor matrix
def predic_mat_init(output_size):
    predic_mat  = [[0 for x in range(3)] for i in range(output_size[1])]
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

def moving_average(data, desired_interval):
    moving_average = np.convolve(data, np.ones(desired_interval), 'same') / desired_interval
    return moving_average

#for now we use offset of just 2 but we will calculate the offset according to the event-related analysis
#the offset is the difference in time between when the stimulus is applied and when we expect to see an action potential since it is not immediate
def predic_mat_gen(read_data, size, predic_mat, cutoff, time_interval):
    for i in range(size[0]):
        predic_mat_size = np.shape(predic_mat)
        if read_data[i][0] == "plus":
            pos_time = read_data[i][1]
            pos_samp = int(pos_time * 5)
            for j in range(cutoff):
                if (j+ lick_samp) < predic_mat_size[0]:
                    predic_mat[pos_samp+j][0] = Gcamp_decay(j * time_interval, 2)
        elif read_data[i][0] == "minus":
            neg_time = read_data[i][1]
            neg_samp = int(neg_time * 5)
            for j in range(cutoff):
                if (j+ lick_samp) < predic_mat_size[0]:
                    predic_mat[neg_samp+j][1] = Gcamp_decay(j * time_interval, 2)
        elif read_data[i][0] == "licks":
            lick_time = read_data[i][1]
            lick_samp = int(lick_time * 5)
            for j in range(cutoff):
                if (j+ lick_samp) < predic_mat_size[0]:
                    predic_mat[lick_samp+j][2] = Gcamp_decay(j * time_interval, 2) 
    return predic_mat
