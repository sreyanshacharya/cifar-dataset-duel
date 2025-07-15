import tensorflow as tf
from tensorflow import keras
import numpy

cifar100 = keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

x_test = x_test/255.0
x_train = x_train/255.0

model = keras.Sequential([
  keras.layers.Conv2D(64, (3,3), input_shape=(32, 32, 3), activation='relu'),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Conv2D(128, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(20, activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))