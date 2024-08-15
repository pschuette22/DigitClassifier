import os
import coremltools as ct
from PIL import Image
import numpy as np
import argparse
from pathlib import Path

def compare(models):
    """Compare the accuracy of multiple models on a validation dataset."""

    if len(models) <= 2:
        raise ValueError("Please provide at least 2 models to compare.")
    validate_images = 'dataset/fonts/validate'

    # Maintian a count of correctly identified images
    # models = [("Apple", apple_model), ("Basic", basic_model), ("Tuned", tuned_model)]
    hits = [0, 0, 0]
    images = 0
    # Iterate over the files in the folder
    for root, _, files in os.walk(validate_images):
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
    for model in models:
        index= models.index(model)
        print(f"{model[0]}: {hits[index]} out of {images}: {hits[index] / images}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process paths to mlmodel files and find concerning fonts.')
    parser.add_argument('-m', '--models', nargs='+', type=str)
    args = parser.parse_args()

    models = []
    for arg in args.models:
        print(arg)
        if os.path.isfile(arg) and arg.endswith('.mlmodel'):
            model_name = Path(arg).stem
            models.append((model_name, ct.models.MLModel(arg)))
        else:
            raise FileNotFoundError(f"The file {arg} is not a valid mlmodel file.")
    if len(models) <= 2:
        raise ValueError("Please provide at least 2 models to compare.")
    compare(models)