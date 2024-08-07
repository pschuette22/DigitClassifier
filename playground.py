from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical


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

fonts_test_dataset = image_dataset_from_directory(
    font_test_dataset,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(28, 28),
    shuffle=False
)

# Normalize the images to the [0, 1] range
normalization_layer = tf.keras.layers.Rescaling(1./255)
fonts_train_dataset = fonts_train_dataset.map(lambda x, y: (normalization_layer(x), y))
fonts_test_dataset = fonts_test_dataset.map(lambda x, y: (normalization_layer(x), y))

# # Convert the labels to categorical
# fonts_train_dataset = fonts_train_dataset.map(lambda x, y: (x, to_categorical(y, 10)))
# fonts_test_dataset = fonts_test_dataset.map(lambda x, y: (x, to_categorical(y, 10)))


# Combine the fonts dataset with the existing MNIST dataset
x_train_fonts = np.concatenate([x for x, y in fonts_train_dataset])
y_train_fonts = np.concatenate([y for x, y in fonts_train_dataset])
y_train_fonts.astype(np.uint8)

x_test_fonts = np.concatenate([x for x, y in fonts_test_dataset])
y_test_fonts = np.concatenate([y for x, y in fonts_test_dataset])
y_test_fonts.astype(np.uint8)

print("Font data example")
i = 200
while i < x_train_fonts.shape[0]:
    print_digit_representation(x_train_fonts[i])
    print(y_train_fonts[i])
    i += 1000

i = 100
while i < x_test_fonts.shape[0]:
    print_digit_representation(x_test_fonts[i])
    print(y_test_fonts[i])
    i += 800

# Print the shapes of the combined datasets
print("Combined x_train shape:", x_train_fonts.shape)
print("Combined y_train shape:", y_train_fonts.shape)
print("Combined x_test shape:", x_test_fonts.shape)
print("Combined y_test shape:", y_test_fonts.shape)