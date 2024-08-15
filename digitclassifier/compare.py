import os
import coremltools as ct
from PIL import Image
import numpy as np
# from IPython.display import display, Markdown, Latex

iteration = 5
apple_model = ct.models.MLModel('MNISTClassifier.mlmodel')
basic_model = ct.models.MLModel(f'product/DigitClassifier{iteration}.mlmodel')
tuned_model = ct.models.MLModel(f'product/TunedDigitClassifier{iteration}.mlmodel')

validate_images = 'dataset/fonts/validate'

# Maintian a count of correctly identified images
models = [("Apple", apple_model), ("Basic", basic_model), ("Tuned", tuned_model)]
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
            prediction = model[1].predict({'image': image})
            digit = int(prediction['classLabel'])
            # Check if the prediction is correct
            if digit == actual_digit:
                hits[models.index(model)] += 1
            else:
                model_misses += 1
        if model_misses == 3:
            print(f"All models missed {file_path}")

print()
print(" === ")
print()
print(f"Apple: {hits[0]} out of {images}: {hits[0] / images}")
print(f"Basic: {hits[1]} out of {images}: {hits[1] / images}")
print(f"Tuned: {hits[2]} out of {images}: {hits[2] / images}")
print()