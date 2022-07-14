import numpy as np

x = [i for i in range(1,11)]
y = [j for j in range(11,21)]

array = np.array([x,y])
print(array.ravel())