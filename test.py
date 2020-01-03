from data.data import process_data
import numpy as np


file1 = 'data/train/'
file2 = 'data/test/'
X_train, y_train, _, _ = process_data(file1, file2)

X_train = np.reshape(X_train, (X_train.shape[0],-1))
print(X_train.shape)
