import tensorflow as tf
from tensorflow import keras
import numpy
import matplotlib
from matplotlib import pyplot as plt

cifar10 = keras.datasets.cifar10
(x10_train, y10_train), (x10_test, y10_test) = cifar10.load_data()

cifar100 = keras.datasets.cifar100
(x100_train, y100_train), (x100_test, y100_test) = cifar100.load_data(label_mode='fine')

x10_test, x100_test, x10_train, x100_train = x10_test/255.0, x100_test/255.0, x10_train/255.0, x100_train/255.0

def buildModel(outputclasses) :
  model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(outputclasses, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

model10 = buildModel(10)
c10fit = model10.fit(x10_train, y10_train, epochs=15, validation_data = (x10_test, y10_test), verbose=2)

model100 = buildModel(100)
c100fit = model100.fit(x100_train, y100_train, epochs=15, validation_data = (x100_test, y100_test), verbose=2)

plt.plot(c10fit.history['accuracy'], label='cifar10 training accuracy')
plt.plot(c10fit.history['val_accuracy'], label='cifar10 validation accuracy')
plt.xlabel('no. of epochs')
plt.ylabel('accuracy metric')
plt.title('Cifar10 Training Graph')
plt.legend()
plt.savefig('c10.png')
plt.close()

plt.plot(c100fit.history['accuracy'], label='cifar100 training accuracy')
plt.plot(c100fit.history['val_accuracy'], label='cifar100 validation accuracy')
plt.xlabel('no. of epochs')
plt.ylabel('accuracy metric')
plt.title('Cifar100 Training Graph')
plt.legend()
plt.savefig('c100.png')
plt.close()


