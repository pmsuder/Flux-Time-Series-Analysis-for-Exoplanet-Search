import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

import sklearn
import tensorflow as tf
from tensorflow import keras

from filterbanks import mel_filter


################# Training Data ##################
df_train = pd.read_csv('data/exoTrain.csv')

n = 200
nfilt = 50
x_train = np.empty((n, nfilt))

y_train = np.array(df_train['LABEL'][0:n]).flatten()

for i in range(n):
        
    arr = df_train.iloc[i].to_numpy()[1:]
    ind = np.arange(1,np.shape(arr)[0]+1)

    FFT = np.real(np.fft.fft(arr))
    FFT = FFT.flatten()
    #plt.plot(ind, FFT)
    #plt.show()

    energies = mel_filer(abs(FFT), min_freq = 0, max_freq = 1000, nfilters = nfilt, nfft = 900)
    #plt.plot(energies)
    #plt.show()
    x_train[i,:] = energies


############### Testing Data ####################

df_test = pd.read_csv('data/exoTest.csv')

n = 20
nfilt = 50
x_test = np.empty((n, nfilt))

y_test = np.array(df_test['LABEL'][0:n]).flatten()


for i in range(n):
        
    arr = df_test.iloc[i].to_numpy()[1:]
    ind = np.arange(1,np.shape(arr)[0]+1)

    FFT = np.real(np.fft.fft(arr))
    FFT = FFT.flatten()

    energies = mel_filer(abs(FFT), min_freq = 0, max_freq = 1000, nfilters = nfilt, nfft = 900)
    x_test[i,:] = energies


###############################################
########## FITTING MODELS #####################
###############################################

def preprocess(x, y):
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.int64)

  return x, y

def create_dataset(xs, ys, n_classes=2):
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(128)


train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_test, y_test)


print(y_train)
print(y_test)


###################################
########## SVM ####################
###################################
clf = sklearn.svm.SVC(kernel = 'linear', C = 2.0)

clf.fit(x_train, y_train)
print(clf.predict(x_test))
input('press any key and Enter to continue ')



#####################################
########### DEEP NEURAL NET #########
#####################################


model = keras.Sequential([
    keras.layers.Reshape(target_shape=(nfilt,), input_shape=(nfilt, )),
    keras.layers.Dense(units=5, activation='relu'),
    keras.layers.Dense(units=2, activation='softmax')
])


#### training
model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_dataset.repeat(), 
    epochs=10, 
    steps_per_epoch=500,
    validation_data=val_dataset.repeat(), 
    validation_steps=2
)

predictions = model.predict(val_dataset)
print(predictions)

