##
import coremltools

def create_keras_base_model(url):
    """This method creates a convolutional neural network model using Keras.
    url - The URL that the keras model will be saved as h5 file.
    """
    
    import keras
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    
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
                  optimizer=keras.optimizers.SGD(learning_rate=0.01),
                  metrics=['accuracy'])

    model.save(url)

keras_model_path = 'product/KerasMnist.h5'
create_keras_base_model(keras_model_path)

def convert_keras_to_mlmodel(keras_url, mlmodel_url):
    """This method simply converts the keras model to a mlmodel using coremltools.
    keras_url - The URL the keras model will be loaded.
    mlmodel_url - the URL the Core ML model will be saved.
    """
    from keras.models import load_model
    keras_model = load_model(keras_url)
    
    # from coremltools.converters import keras as keras_converter
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    classifier_config = coremltools.ClassifierConfig(class_labels)

    mlmodel = coremltools.converters.convert(
        keras_model,
        source="tensorflow",
        convert_to="neuralnetwork",
        inputs=[coremltools.TensorType(shape=(1, 784))],
        classifier_config=classifier_config
        )
    mlmodel.save(mlmodel_url)
     
coreml_model_path = 'product/KerasMNISTDigitClassifier.mlmodel'
convert_keras_to_mlmodel(keras_model_path , coreml_model_path)

spec = coremltools.utils.load_spec(coreml_model_path)
builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)
builder.inspect_layers(last=3)

# let's inspect the input of the model as we need this information later on the make_updatable method
builder.inspect_input_features()

neuralnetwork_spec = builder.spec

# change the input so the model can accept 28x28 grayscale images
neuralnetwork_spec.description.input[0].type.imageType.width = 28
neuralnetwork_spec.description.input[0].type.imageType.height = 28

from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
grayscale = _FeatureTypes_pb2.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
neuralnetwork_spec.description.input[0].type.imageType.colorSpace = grayscale

# let's inspect the input again to confirm the change in input type
builder.inspect_input_features()

# Set input and output description
neuralnetwork_spec.description.input[0].shortDescription = 'Input image of the handwriten digit to classify'
neuralnetwork_spec.description.output[0].shortDescription = 'Probabilities / score for each possible digit'
neuralnetwork_spec.description.output[1].shortDescription = 'Predicted digit'

# Provide metadata
neuralnetwork_spec.description.metadata.author = 'Core ML Tools'
neuralnetwork_spec.description.metadata.license = 'MIT'
neuralnetwork_spec.description.metadata.shortDescription = (
		'An updatable hand-written digit classifier setup to train or be fine-tuned on MNIST like data.')

def make_updatable(builder, mlmodel_url, mlmodel_updatable_path):
    """This method makes an existing non-updatable mlmodel updatable.
    mlmodel_url - the path the Core ML model is stored.
    mlmodel_updatable_path - the path the updatable Core ML model will be saved.
    """
    model_spec = builder.spec

    # make_updatable method is used to make a layer updatable. It requires a list of layer names.
    # dense_1 and dense_2 are two innerProduct layer in this example and we make them updatable.
    builder.make_updatable(['dense_1', 'dense_2'])

    # Categorical Cross Entropy or Mean Squared Error can be chosen for the loss layer.
    # Categorical Cross Entropy is used on this example. CCE requires two inputs: 'name' and 'input'.
    # name must be a string and will be the name associated with the loss layer
    # input must be the output of a softmax layer in the case of CCE. 
    # The loss's target will be provided automatically as a part of the model's training inputs.
    builder.set_categorical_cross_entropy_loss(name='lossLayer', input='digitProbabilities')

    # in addition of the loss layer, an optimizer must also be defined. SGD and Adam optimizers are supported.
    # SGD has been used for this example. To use SGD, one must set lr(learningRate) and batch(miniBatchSize) (momentum is an optional parameter).
    from coremltools.models.neural_network import SgdParams
    builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=32))

    # Finally, the number of epochs must be set as follows.
    builder.set_epochs(10)
        
    # Set training inputs descriptions
    model_spec.description.trainingInput[0].shortDescription = 'Example image of handwritten digit'
    model_spec.description.trainingInput[1].shortDescription = 'Associated true label (digit) of example image'

    # save the updated spec
    from coremltools.models import MLModel
    mlmodel_updatable = MLModel(model_spec)
    mlmodel_updatable.save(mlmodel_updatable_path)

coreml_updatable_model_path = './product/UpdatableMNISTDigitClassifier.mlmodel'
make_updatable(builder, coreml_model_path, coreml_updatable_model_path)

spec = coremltools.utils.load_spec(coreml_updatable_model_path)
builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)

builder.inspect_loss_layers()

# let's inspect the optimizer of the Core ML model
builder.inspect_optimizer()

# let's see which layers are updatable
builder.inspect_updatable_layers()