import pandas as pd
import numpy as np
import os

practice_data = os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6/event_times_VJ_OFCVTA_7_260_D6_trained.csv')
read_data = pd.read_csv(practice_data)
read_data = pd.DataFrame.to_numpy(read_data)
#read_data = read_data.astype(np.array)
hertz = 5 #measurements per second
interval = 0.2 #1/hertz gives us the seconds per measurement
size = np.shape(read_data)
frequency = 5
time_interval = round(1/(frequency), 1)

#initializing predictor matrix
time_total = 0
for i in range(size[0]):
    if read_data[i][1] > time_total:
        time_total = read_data[i][1] 
samp_num = int(time_total * 5)
predic_mat  = [[0 for x in range(3)] for i in range(samp_num)]
predic_mat_size = np.shape(predic_mat)

#initializing indentity-like matrix
# this matrix takes a long time to initialize (~5-10 seconds)
#I  = [[0 for x in range(samp_num * 2)] for i in range(samp_num)]

#read_data[i][1] * 5 = sample #
#we ignore licks for now
#at each sample time we know that the gcamp has been activated and therefore want to model subsequent rows based on the activity of the gcamp
#this activity will be decreasing following a known function of gcamp decay (action potential signal) 
#the point at which we decide the gcamp activity is so small we count it as 0 could impact the error (maybe helps us measure the error)

#creates matrix which indicates at which times CV+ and CV- are applied
cutoff = 10 #number of samples we wish to continue seeing the gcamp after the initial stimulus (cutoff of 10 samples at 5hz is 2 seconds)
for i in range(size[0]):
    if read_data[i][0] == "plus":
        pos_time = read_data[i][1]
        pos_samp = int(pos_time * 5)
        predic_mat[pos_samp][1] = 1
        for j in range(cutoff):
            if pos_samp + j <=  predic_mat_size[0]:
                predic_mat[pos_samp+j][1] = 1
                # future improvement: when we figure out the function for modeling gcamp decay, 
                # we will not just plug in 1, but a calculated value of the magnitude of the expression of gcamp at that time
                # this function will probably depend on j which is the time (each j = time_interval seconds)
    elif read_data[i][0] == "minus":
        neg_time = read_data[i][1]
        neg_samp = int(neg_time * 5)
        predic_mat[neg_samp][2] = 1
        for j in range(cutoff):
            if neg_samp + j <  predic_mat_size[0]:
                predic_mat[neg_samp+j][2] = 1 
                # future improvement: when we figure out the function for modeling gcamp decay, 
                # we will not just plug in 1, but a calculated value of the magnitude of the expression of gcamp at that time
                # this function will probably depend on j which is the time (each j = time_interval seconds)

for i in range(predic_mat_size[0]):
    predic_mat[i][0] = i

print(predic_mat)
