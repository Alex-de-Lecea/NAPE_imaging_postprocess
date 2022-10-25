import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import statsmodels.api as sm

regress_mat = predic_mat_gen(read_data, size, predic_mat, cutoff, time_interval)

#regress_mat = sm.add_constant(regress_mat)
olsmod = sm.OLS(output_data[0], regress_mat)
olsres = olsmod.fit()

ypred = olsres.predict(regress_mat)

pos_stim_arr = np.transpose(regress_mat)[0]
neg_stim_arr = np.transpose(regress_mat)[0]
print(pos_stim_arr)

#plotting pos_stim_arr over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("CS+ expected fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
plt.plot(x, pos_stim_arr, color = "red", marker = "o")
plt.legend()
plt.show()

#plotting neg_stim_arr over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("CS- expected fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
plt.plot(x, neg_stim_arr, color = "red", marker = "o")
plt.legend()
plt.show()

#plotting actual fluoresence over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("Actual fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
plt.plot(x, output_data[0], color = "red", marker = "o")
plt.legend()
plt.show()

#plotting expected fluoresence over time (samples)
x = np.array(range(0, predic_mat_size[0]))
plt.title("Expected fluoresence")
plt.xlabel("samples")
plt.ylabel("Fluoresence")
plt.plot(x, ypred, color = "red", marker = "o")
plt.legend()
plt.show()
