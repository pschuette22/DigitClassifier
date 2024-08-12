import os
import coremltools as ct
from PIL import Image
import numpy as np
# from IPython.display import display, Markdown, Latex

iteration = 11
apple_model = ct.models.MLModel('MNISTClassifier.mlmodel')
basic_model = ct.models.MLModel(f'product/DigitClassifier{iteration}.mlmodel')
tuned_model = ct.models.MLModel(f'product/TunedDigitClassifier{iteration}.mlmodel')

validate_images = 'dataset/fonts/validate'

# Maintian a count of correctly identified images
models = [("Apple", 'image', apple_model), ("Basic", 'input_1', basic_model), ("Tuned", 'input_1', tuned_model)]
hits = [0, 0, 0]
images = 0
tuned_misses = []
missed_digits = np.repeat(0, 10)
guessed_digits = np.repeat(0, 10)
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
                model_misses += 1
                if model[0] == "Tuned":
                    missed_digits[actual_digit] += 1
                    guessed_digits[digit] += 1                    
                    print(f"{model[0]} predicted {digit} incorrectly ({actual_digit}) ({file_path})")
                    missline = file_path
                    if model_misses == 3:
                        missline += f" (Universal miss) {actual_digit} != {digit}"
                    tuned_misses.append(missline)
        if model_misses == 3:
            print(f"All models missed {file_path}")

print()
print(" == Tuned Model Misses == ")
print()

for tuned_miss in tuned_misses:
    print(tuned_miss)
print()
print(" === ")
print()
print(f"Apple: {hits[0]} out of {images}: {hits[0] / images}")
print(f"Basic: {hits[1]} out of {images}: {hits[1] / images}")
print(f"Tuned: {hits[2]} out of {images}: {hits[2] / images}")
print()
for i in range(10):
    print(f"Guessed {i}: {guessed_digits[i]}")
    print(f"Missed {i}: {missed_digits[i]}")