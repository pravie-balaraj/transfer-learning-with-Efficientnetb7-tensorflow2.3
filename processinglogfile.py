import os
import glob
import matplotlib.pyplot as plt
import re

with open('efficientnetB7.log','r') as f:
    data = f.readline()
    while data is not None:
        if re.match(".*val_accuracy.*",data):
            print(data)
        data = f.readline()


import numpy as np
import matplotlib.pyplot as plt

data = np.array([[0.9709,2.4476]])
xaxis_data = np.arange(data.shape[0])
xaxis_data = xaxis_data + 1
fig,ax = plt.subplots(nrows=1,ncols=1)
ax.plot(xaxis_data,data[:,0], label='Training Acc')
ax.plot(xaxis_data,data[:,1], label='Validation Acc')
ax.legend(loc='best')