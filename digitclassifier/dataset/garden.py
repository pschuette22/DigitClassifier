import os
import coremltools as ct
from PIL import Image
from pathlib import Path
import numpy as np
import argparse


def font_digit(file_path) -> int:
    """Return the degit depicted in an image given it's path."""
    return int(file_path.split('/')[-2])

def font_family(file_path) -> str:
    """Return the font family of an image given it's path."""
    return Path(file_path).stem

def print_array_line_by_line(arr):
    """Print each element of the array line by line."""
    for element in arr:
        print(element)

def garden_fonts(models): # (apple_model, basic_model, tuned_model):
    """Identify the fonts that may be so unique they are not valuable training data."""
    font_images = 'dataset/fonts'
    # Maintian a count of correctly identified images
    # models = [("Apple", apple_model), ("Basic", basic_model), ("Tuned", tuned_model)]
    hits = np.repeat(0, 3)
    images = 0
    # Iterate over the files in the folder
    exclusion_fonts = set()
    universal_misses = []
    missed_fonts = {}
    universal_miss_digits = np.repeat(0, 10)
    for root, _, files in os.walk(font_images):
        for file in files:
            if not file.endswith('.png') and not file.endswith('.jpg'):
                # Not an image
                continue
            file_path = os.path.join(root, file)
            actual_digit = font_digit(file_path)
            # Load the image in grayscale
            image = Image.open(file_path).convert('L')
            images += 1
            model_misses = 0
            for model in models:
                # Make a prediction
                prediction = model[1].predict({'image': image})
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
                    print(f"{model[0]} predicted {digit} incorrectly ({file_path})")
                    model_misses += 1
                    missed_fonts[font_family(file_path)] = missed_fonts.get(font_family(file_path), 0) + 1

            #
            # If all models missed, we have a universal miss
            #
            if model_misses == len(models):
                if actual_digit == 9:
                    # Bias towards 9 confusion - this digit in particular causes issues
                    confusion_font = font_family(file_path)
                    exclusion_fonts.add(confusion_font)
                else:
                    universal_miss_digits[actual_digit] += 1
                    universal_misses.append(file_path)

                print(f"All models missed {file_path}")

    print(f"Apple: {hits[0]} out of {images}: {hits[0] / images}")
    print(f"Basic: {hits[1]} out of {images}: {hits[1] / images}")
    print(f"Tuned: {hits[2]} out of {images}: {hits[2] / images}")
    print("-- Universal misses --")
    print(f"Total misses: {len(universal_misses)}")
    print_array_line_by_line(universal_misses)

    # Determine what universal misses cover the entire font family

    font_family_counts = {}
    for font_file in universal_misses:
        missed_font = font_family(font_file)
        digit = font_digit(font_file)
        font_family_counts[missed_font] = font_family_counts.get(missed_font, 0) + 1
        if font_family_counts[missed_font] >= 3: # All models missed on 3 digits
            exclusion_fonts.add(missed_font)

    for missed_font in missed_fonts:
        if missed_fonts[missed_font] >= 12: # 40% miss with this font
            exclusion_fonts.add(missed_font)

    print(f"Total exclusion fonts: {len(exclusion_fonts)}")

    # Prune and write the universal misses to our exclusion file
    file = open('dataset/ignored.txt', mode='a')
    for excluded in exclusion_fonts:
        print(f"Writing {excluded} to ignore file")
        file.write('\n')
        file.write(excluded)

    file.close()

apple_model = ct.models.MLModel('MNISTClassifier.mlmodel')
basic_model = ct.models.MLModel('product/DigitClassifier3.mlmodel')
tuned_model = ct.models.MLModel('product/TunedDigitClassifier3.mlmodel')

garden_fonts(apple_model, basic_model, tuned_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process path to a directory containing font files')
    parser.add_argument('directory', type=str, help='The path to the google fonts project directory')
    args = parser.parse_args()
    if os.path.isdir(args.directory):
        build_dataset(args.directory)
    else:
        raise FileNotFoundError(f"The directory ${args.directory} doesnt exist: pass a valid path to the google fonts repository.")