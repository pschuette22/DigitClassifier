import os
import coremltools as ct
from PIL import Image
from pathlib import Path

apple_model = ct.models.MLModel('MNISTClassifier.mlmodel')
basic_model = ct.models.MLModel('product/DigitClassifier.mlmodel')
tuned_model = ct.models.MLModel('product/TunedDigitClassifier.mlmodel')

font_images = 'dataset/fonts'

# Maintian a count of correctly identified images
models = [("Apple", 'image', apple_model), ("Basic", 'input_1', basic_model), ("Tuned", 'input_1', tuned_model)]
hits = [0, 0, 0]
images = 0
# Iterate over the files in the folder
universal_misses = []
for root, dir, files in os.walk(font_images):
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
            print(model[0])
            prediction = model[2].predict({model[1]: image})
            print(prediction)
            digit = int(prediction['classLabel'])
            confidence = 0
            if model[0] == "Apple":
                confidence = prediction["labelProbabilities"][digit]
            else:
                confidence = prediction["Identity"][str(digit)]
            # Check if the prediction is correct and we are confident in it
            if digit == actual_digit and confidence > 0.3:
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
            universal_misses.append(file_path)
            print(f"All models missed {file_path}")

def print_array_line_by_line(arr):
    """Print each element of the array line by line."""
    for element in arr:
        print(element)

print(f"Apple: {hits[0]} out of {images}: {hits[0] / images}")
print(f"Basic: {hits[1]} out of {images}: {hits[1] / images}")
print(f"Tuned: {hits[2]} out of {images}: {hits[2] / images}")
print("-- Universal misses --")
print(f"Total misses: {len(universal_misses)}")
print_array_line_by_line(universal_misses)

# Determine what universal misses cover the entire font family
universal_miss_fonts = set()
font_family_counts = {'some': 0}
for missed_font in universal_misses:
    image_path_stem = Path(missed_font).stem
    image_name = image_path_stem.split('/')[-1]
    digit = int(image_name.split("_")[0])
    font_name = image_name.replace(f"{digit}_", "")
    if digit == 0:
        continue # Ignore 0, false positives come from an empty box
    if font_family_counts.get(font_name) is None:
        font_family_counts[font_name] = 1
    elif font_family_counts[font_name] >= 4: # Over half are missed
        universal_miss_fonts.add(font_name)
    else:
        font_family_counts[font_name] += 1

print(f"Total universal miss fonts: {len(universal_miss_fonts)}")

# Prune and write the universal misses to our exclusion file
file = open('dataset/ignored.txt', mode='a')
for missed_font in universal_miss_fonts:
    print(f"Writing {missed_font} to ignore file")
    file.write('\n')
    file.write(missed_font)

file.close()
