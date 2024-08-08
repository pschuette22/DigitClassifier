import os
import coremltools as ct
from PIL import Image
# from IPython.display import display, Markdown, Latex

apple_model = ct.models.MLModel('MNISTClassifier.mlmodel')
basic_model = ct.models.MLModel('product/DigitClassifier1.mlmodel')
tuned_model = ct.models.MLModel('product/TunedDigitClassifier1.mlmodel')

validate_images = 'dataset/fonts/validate'

# Maintian a count of correctly identified images
models = [("Apple", 'image', apple_model), ("Basic", 'input_1', basic_model), ("Tuned", 'input_1', tuned_model)]
hits = [0, 0, 0]
images = 0
# Iterate over the files in the folder
for root, dir, files in os.walk(validate_images):
    for file in files:
        if not file.endswith('.png') and not file.endswith('.jpg'):
            # Not an image
            continue
        actual_digit = int(root.split('/')[-1])
        file_path = os.path.join(root, file)
        
        # Load the image in grayscale
        image = Image.open(file_path).convert('L')
        images += 1
        model_misses = 0
        for model in models:
            # Make a prediction
            prediction = model[2].predict({model[1]: image})
            digit = int(prediction['classLabel'])
            # Check if the prediction is correct
            if digit == actual_digit:
                hits[models.index(model)] += 1
                # print(f"{model[0]} predicted {digit} correctly")
            else:
                # Get the predicted digit
                # print(prediction)
                print(f'Predicted: {digit}')
                # Get the actual digit
                print(f'Actual: {actual_digit}')
                print(f"{model[0]} predicted {digit} incorrectly ({file_path})")
                model_misses += 1
        if model_misses == 3:
            print(f"All models missed {file_path}")

print(f"Apple: {hits[0]} out of {images}: {hits[0] / images}")
print(f"Basic: {hits[1]} out of {images}: {hits[1] / images}")
print(f"Tuned: {hits[2]} out of {images}: {hits[2] / images}")