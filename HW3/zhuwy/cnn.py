from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
import numpy as np 
X_train = np.load('X_train.npy')[:,:,:,np.newaxis] # (N,28,28,1)
Y_train = np.load('Y_train.npy')[:,np.newaxis] # (N,1)
Y_train = to_categorical(Y_train)
print(X_train.shape,Y_train.shape)

X_test = np.load('X_test.npy')[:,:,:,np.newaxis] # (N,28,28,1)
Y_test = np.load('Y_test.npy')[:,np.newaxis] # (N,1)
Y_test = to_categorical(Y_test)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(BatchNormalization(input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Y_test.shape[-1], activation='softmax'))

adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

model.fit(X_train, Y_train, batch_size=32, epochs=3)
test_loss,test_acc = model.evaluate(X_test, Y_test, batch_size=32)
print('Test loss:',test_loss,'\nTest accuary:',test_acc)
plot_model(model, to_file='simlpecnn.png',show_shapes=True)