import tensorflow as tf
import numpy
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


cifar10 = keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test=x_test/255.0
x_train=x_train/255.0

model = keras.Sequential([
  keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2, 2),
  keras.layers.Flatten(),
  keras.layers.Dense(256, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_test, 
                    y_test,
                    epochs=16,
                    validation_data = (x_test, y_test))



plt.plot(history.history['accuracy'], label='training acc.')
plt.plot(history.history['val_accuracy'], label='validation acc.')
plt.legend()
plt.show()
