from PIL import Image
import os
import numpy as np 
X = [] # X (N,28,28)
Y = [] # Y (N,)
for root,parent,files in os.walk('Train'):
    for _file in files:
        img = Image.open(os.path.join(root,_file))
        X.append(np.array(img))
        label = _file.split('.')[0][-1]
        Y.append(ord(label)-65)
X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape,Y.shape)
np.save('X_train.npy',X)
np.save('Y_train.npy',Y)

X = []
Y = []
for root,parent,files in os.walk('Test'):
    for _file in files:
        img = Image.open(os.path.join(root,_file))
        X.append(np.array(img))
        label = _file.split('.')[0][-1]
        Y.append(ord(label)-65)
X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape,Y.shape)
np.save('X_test.npy',X)
np.save('Y_test.npy',Y)