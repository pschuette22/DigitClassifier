import os
import numpy as np
import keras
from keras import layers
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def print_digit_representation(representation):
    for x in representation:
        row = ""
        for y in x:
            if y > 0.6:
                row += "X"
            elif y > 0.3:
                row += "-"
            else:
                row += " "
        print(row)

def ensure_unique(file_path: str):
    """Adds a unique index, to a filepath so ensure no overwrites
    """
    from pathlib import Path
    unique_path = file_path
    # TODO: make this more robust or find the proper lib (Low priority)
    # does not work with directories containing '.'
    stem = file_path.split('.')[0]
    extension = file_path.removeprefix(f"{stem}.")
    index = 0
    while os.path.isfile(unique_path):
        index += 1
        unique_path = f"{stem}{index}.{extension}"
    return unique_path


def convert_keras_to_mlmodel(keras_model_url, mlmodel_url):
    """This method simply converts the keras model to a mlmodel using coremltools.
    keras_url - The URL the keras model will be loaded.
    mlmodel_url - the URL the Core ML model will be saved.
    """
    from keras.models import load_model
    import coremltools as ct
    
    # from coremltools.converters import keras as keras_converter
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    classifier_config = ct.ClassifierConfig(class_labels)
    keras_model = load_model(keras_model_url)
    mlmodel = ct.converters.convert(
        keras_model,
        source="tensorflow",
        convert_to="neuralnetwork",
        inputs=[ct.ImageType(shape=(1, 28, 28, 1), color_layout=ct.colorlayout.GRAYSCALE)],
        classifier_config=classifier_config
    )
    mlmodel.save(mlmodel_url)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Import font data
font_training_data = 'dataset/fonts/train'
font_test_dataset = 'dataset/fonts/test'

# Load the dataset from the fonts directory
fonts_train_dataset = image_dataset_from_directory(
    font_training_data,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(28, 28),
    shuffle=False
)
fonts_train_dataset.shuffle(100)

fonts_test_dataset = image_dataset_from_directory(
    font_test_dataset,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(28, 28),
    shuffle=False
)
fonts_test_dataset.shuffle(100)

# Normalize the images to the [0, 1] range
normalization_layer = tf.keras.layers.Rescaling(1./255)
fonts_train_dataset = fonts_train_dataset.map(lambda x, y: (normalization_layer(x), y))
fonts_test_dataset = fonts_test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Split to format model fitting
x_train_fonts = np.concatenate([x for x, y in fonts_train_dataset])
y_train_fonts = np.concatenate([y for x, y in fonts_train_dataset])
y_train_fonts.astype(np.uint8)

x_test_fonts = np.concatenate([x for x, y in fonts_test_dataset])
y_test_fonts = np.concatenate([y for x, y in fonts_test_dataset])
y_test_fonts.astype(np.uint8)

print("MNIST data example")
print_digit_representation(x_train[0])
print(y_train[0])

batch_size = 32
epochs = 4

keras.backend.clear_session()
model = keras.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9),
                metrics=['accuracy'])

model.summary()

model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)

mnist_score = model.evaluate(x_test, y_test, verbose=0)
print("MNIST Test loss:", mnist_score[0])
print("MNIST Test accuracy:", mnist_score[1])
font_score = model.evaluate(x_test_fonts, y_test_fonts, verbose=0)
print("Font Test loss:", font_score[0])
print("Font Test accuracy:", font_score[1])

model.summary()

# Save the model

keras_model_path = ensure_unique('product/mnist_model.h5')
model.save(keras_model_path)
digit_classifier_path = ensure_unique('product/DigitClassifier.mlmodel')
convert_keras_to_mlmodel(keras_model_path, digit_classifier_path)

# Continue with the rest of your code to train the model
model.fit(
    x_train_fonts, y_train_fonts, batch_size=64, epochs=2, validation_split=0.1
)

mnist_score = model.evaluate(x_test, y_test, verbose=0)
print("MNIST Test loss:", mnist_score[0])
print("MNIST Test accuracy:", mnist_score[1])
font_score = model.evaluate(x_test_fonts, y_test_fonts, verbose=0)
print("Font Test loss:", font_score[0])
print("Font Test accuracy:", font_score[1])

model.summary()

# Save the model
tuned_keras_model_path = ensure_unique('product/tuned_mnist_model.h5')
model.save(tuned_keras_model_path)
tuned_digit_classifier_path = ensure_unique('product/TunedDigitClassifier.mlmodel')
convert_keras_to_mlmodel(tuned_keras_model_path, tuned_digit_classifier_path)