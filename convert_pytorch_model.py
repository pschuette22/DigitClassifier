#
# Convert a build .pt model to CoreML and make it updatable
#
import torch
from torch import nn
import coremltools

#
# Import the loaded model
#

model = torch.load('./product/digit_classifier.pt')
tracable = model.eval()
random = torch.rand(1, 784)
print(" == Tracing params ==")
print(tracable)
print(" == Random params == ")
print(random)
print(random.shape)
traced = torch.jit.trace(tracable, random)

# Using image_input in the inputs parameter:
# Convert to Core ML using the Unified Conversion API.

converted_model = coremltools.convert(
    traced,
    source="pytorch",
    inputs=[coremltools.TensorType(shape=(1, 784))],
    convert_to="mlprogram"
)

#
# == Make the model updatable == 
#

converted_model.user_defined_metadata['com.apple.coreml.model.updatable'] = 'true'

# Save the converted model.
updatable_model_package = "product/UpdatableMNISTClassifier.mlpackage"
converted_model.save(updatable_model_package)
spec = coremltools.utils.load_spec(updatable_model_package)

# Ensure the spec is not None
if spec is None:
    raise ValueError("Failed to load the model specification.")

print(spec)

builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)

# Define the update parameters
# update_params = coremltools.models.neural_network.UpdateParameters(
#     optimizer=coremltools.models.neural_network.SGDOptimizer(learning_rate=0.01),
#     loss=coremltools.models.neural_network.CategoricalCrossEntropyLoss(),
#     epochs=10,
#     batch_size=32
# )

# Add the update parameters to the model
# builder.set_update_parameters(update_params)
coremltools.models.utils.save_spec(builder.spec, updatable_model_package)
print("Model successfully converted, made updatable, and saved.")