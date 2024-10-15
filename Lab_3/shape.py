import numpy as np

ytrain1 = np.load('ytrain1.npy')

crater_count = 0
no_crater_count = 0

for i in ytrain1:
    if i == 1:
        crater_count += 1
    else:
        no_crater_count += 1


print('crater_count: ', crater_count)
print('no_crater_count: ', no_crater_count)