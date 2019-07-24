from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import emnist
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

X_train, Y_train = emnist.extract_training_samples('balanced')
X_test, Y_test = emnist.extract_test_samples('balanced')

 
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
X_train = X_train[:-1]
Y_train = Y_train[:-1]

print(X_train[0].shape)

print(X_train.shape)
# X_train = X_train[:10000]
# Y_train = Y_train[:10000]

X_val = X_test[:len(X_test)//2]
X_test = X_test[len(X_test)//2:]

Y_val = Y_test[:len(Y_test)//2]
Y_test = Y_test[len(Y_test)//2:]

X_train = np.expand_dims(X_train, axis=3)
X_val = np.expand_dims(X_val, axis=3)
X_test = np.expand_dims(X_test, axis=3)

Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
Y_test = to_categorical(Y_test)

input_shape = (28, 28, 1)
num_classes = 47

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
# num_classes = 10

epochs = 30

# model.fit(X_train, Y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(X_val, Y_val))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# model.save('EMNIST_CNN_ACC_' + str(round(score[1], 4))  + '.h5')