from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np 

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

np.random.seed(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


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
# tensorboard = TensorBoard(log_dir='logs/')
# model.fit(X_train, Y_train, batch_size=32, epochs=30,validation_data=(X_test,Y_test),callbacks=[tensorboard])
TR_LOG = []
VAL_LOG = []
TE_LOG = []
for i in range(30):
    HIS = model.fit(X_train, Y_train, batch_size=32, epochs=1,validation_split=0.2)
    tr_loss = HIS.history['loss']
    tr_acc = HIS.history['accuracy']
    tr_precision = HIS.history['precision']
    tr_recall = HIS.history['recall']
    tr_f1 = HIS.history['fmeasure']
    TR_LOG.append(np.array([tr_loss,tr_acc,tr_precision,tr_recall,tr_f1]))

    val_loss = HIS.history['val_loss']
    val_acc = HIS.history['val_accuracy']
    val_precision = HIS.history['val_precision']
    val_recall = HIS.history['val_recall']
    val_f1 = HIS.history['val_fmeasure']
    VAL_LOG.append(np.array([val_loss,val_acc,val_precision,val_recall,val_f1]))

    # print(model.evaluate(X_test, Y_test, batch_size=32))
    te_loss,te_acc,te_precision,te_recall,te_f1 = model.evaluate(X_test, Y_test, batch_size=32)
    TE_LOG.append(np.array([te_loss,te_acc,te_precision,te_recall,te_f1]))

# print('Test loss:',test_loss,'\nTest accuary:',test_acc)
# plot_model(model, to_file='simlpecnn.png',show_shapes=True)

TR_LOG = np.asarray(TR_LOG)
VAL_LOG = np.asarray(VAL_LOG)
TE_LOG = np.asarray(TE_LOG)
print(TR_LOG.shape,TE_LOG.shape)
# np.save('logs.npy',np.array([TR_LOG,VAL_LOG,TE_LOG]))

fig = plt.figure()
x = np.arange(30) + 1.
plt.plot(x,TR_LOG[:,0],label='tr_loss')
plt.plot(x,VAL_LOG[:,0],label='val_loss')
plt.plot(x,TE_LOG[:,0],label='te_loss')
plt.legend()
plt.savefig('loss.png')
plt.clf()

plt.plot(x,TR_LOG[:,1],label='tr_acc')
plt.plot(x,VAL_LOG[:,1],label='val_acc')
plt.plot(x,TE_LOG[:,1],label='te_acc')
plt.legend()
plt.savefig('acc.png')
plt.clf()

plt.plot(x,TR_LOG[:,2],label='tr_precision')
plt.plot(x,VAL_LOG[:,2],label='val_precision')
plt.plot(x,TE_LOG[:,2],label='te_precision')
plt.legend()
plt.savefig('precision.png')
plt.clf()

plt.plot(x,TR_LOG[:,3],label='tr_recall')
plt.plot(x,VAL_LOG[:,3],label='val_recall')
plt.plot(x,TE_LOG[:,3],label='te_recall')
plt.legend()
plt.savefig('recall.png')
plt.clf()

plt.plot(x,TR_LOG[:,4],label='tr_f1')
plt.plot(x,VAL_LOG[:,4],label='val_f1')
plt.plot(x,TE_LOG[:,4],label='te_f1')
plt.legend()
plt.savefig('f1.png')
plt.clf()