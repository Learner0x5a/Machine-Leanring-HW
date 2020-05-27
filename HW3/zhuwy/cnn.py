from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np 
np.random.seed(1)

X_train = np.load('X_train.npy')[:,:,:,np.newaxis] # (N,28,28,1)
Y_train = np.load('Y_train.npy')[:,np.newaxis] # (N,1)
Y_train = to_categorical(Y_train)
print(X_train.shape,Y_train.shape)

X_test = np.load('X_test.npy')[:,:,:,np.newaxis] # (N,28,28,1)
Y_test = np.load('Y_test.npy')[:,np.newaxis] # (N,1)
Y_test = to_categorical(Y_test)
print(X_test.shape,Y_test.shape)

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
 
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

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
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy",precision, recall, fmeasure])
tensorboard = TensorBoard(log_dir='logs/')
model.fit(X_train, Y_train, batch_size=32, epochs=30,validation_data=(X_test,Y_test),callbacks=[tensorboard])
# test_loss,test_acc = model.evaluate(X_test, Y_test, batch_size=32)
# print('Test loss:',test_loss,'\nTest accuary:',test_acc)
# plot_model(model, to_file='simlpecnn.png',show_shapes=True)