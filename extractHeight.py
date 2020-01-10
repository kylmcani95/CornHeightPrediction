import numpy as np


data = np.genfromtxt('video_train.csv', delimiter=',', dtype='str')

data = data[1:, 2]

data = data.astype(int)

np.savetxt('height.csv', data)

