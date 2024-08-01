#
# Train MNIST Digit Classifier using Keras
# https://apple.github.io/coremltools/docs-guides/source/updatable-neural-network-classifier-on-mnist-dataset.html
#

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import coremltools
from coremltools.converters import keras as keras_converter

# Create the base model
keras.backend.clear_session()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr=0.01),
                metrics=['accuracy'])

keras_model_path = 'product/keras_mnist.h5'
model.save(keras_model_path)

# TODO absract to subscript
keras_model = load_model(keras_model_path)