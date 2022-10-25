import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import statsmodels.api as sm

#initializing predictor matrix
def predic_mat_init(output_size):
    predic_mat  = [[0 for x in range(2)] for i in range(output_size[1])]
    return predic_mat

#Gcamp decay function
def Gcamp_decay(time, tau):
    output = (1/tau) * np.exp(-time/tau)
    return output


def predic_mat_gen(read_data, size, predic_mat, cutoff, time_interval):
    for i in range(size[0]):
        if read_data[i][0] == "plus":
            pos_time = read_data[i][1]
            pos_samp = int(pos_time * 5)
            for j in range(cutoff):
                predic_mat[pos_samp+j][0] = round(Gcamp_decay(j * time_interval, 2), 3)
        elif read_data[i][0] == "minus":
            neg_time = read_data[i][1]
            neg_samp = int(neg_time * 5)
            for j in range(cutoff):
                predic_mat[neg_samp+j][1] = round(Gcamp_decay(j * time_interval, 2), 3)
    return predic_mat
