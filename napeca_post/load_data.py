import scipy.io
import numpy as np

mat = scipy.io.loadmat(r'C:\Users\Alex_dl\Documents\GitHub\NAPE_imaging_postprocess\napeca_post\sample_data\PL01\o1d1\Fall')
mat = np.array(mat)

print(np.shape(mat))
print(mat)