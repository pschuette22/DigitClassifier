from keras.datasets import mnist
from keras import backend as K

def mnist_data():
# input image dimensions    
    img_rows, img_cols = 28, 28
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':        
          X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)        
          X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)        
          input_shape = (1, img_rows, img_cols)    
    else:        
          X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)        
          X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)        
          input_shape = (img_rows, img_cols, 1)

    # rescale [0,255] --> [0,1]    
    X_train = X_train.astype('float32')/255    
    X_test = X_test.astype('float32')/255

    # transform to one hot encoding    
    Y_train = np_utils.to_categorical(Y_train, 10)    
    Y_test = np_utils.to_categorical(Y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)

(X_train, Y_train), (X_test, Y_test) = mnist_data()

# plot first six training images
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.cm as cm
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig = plt.figure(figsize=(20,20))

for i in range(6):    
      ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])    
      ax.imshow(X_train[i], cmap='gray')    
      ax.set_title(str(y_train[i]))

# visualize one number with pixel values
def visualize_input(img, ax):    
      ax.imshow(img, cmap='gray')    
      width, height = img.shape    
      thresh = img.max()/2.5    
      for x in range(width):        
          for y in range(height):            
                 ax.annotate(str(round(img[x][y],2)), xy=(y,x),                        
                             horizontalalignment='center',                   
                             verticalalignment='center',                        
                             color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(X_train[0], ax)

# defining the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
def network():    
     model = Sequential()    
     input_shape = (28, 28, 1)    
     num_classes = 10

     model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))    
     model.add(MaxPooling2D(pool_size=2))    
     model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))    
     model.add(MaxPooling2D(pool_size=(2, 2)))    
     model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))    
     model.add(MaxPooling2D(pool_size=(2, 2)))    
     model.add(Dropout(0.3))    
     model.add(Flatten())    
     model.add(Dense(500, activation='relu'))    
     model.add(Dropout(0.4))    
     model.add(Dense(num_classes, activation='softmax'))

     # summarize the model    
     # model.summary()    
     return model

#Training the model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=512, epochs=6, verbose=1,validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

coreml_model = coremltools.converters.keras.convert(model,                                                   
                                                    input_names="image",
                                                    image_input_names='image',
                                                    class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

#entering metadata
coreml_model.author = 'plotti'
coreml_model.license = 'MIT'
coreml_model.short_description = 'MNIST handwriting recognition with a 3 layer network'
coreml_model.input_description['image'] = '28x28 grayscaled pixel values between 0-1'
coreml_model.save('SimpleMnist.mlmodel')

print(coreml_model)

