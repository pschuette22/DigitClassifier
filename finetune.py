#!/usr/bin/env python3

# Fine tune the MNIST dataset to focus on computer fonts

import os
import coremltools as ct
import turicreate as tc
import numpy as np
from PIL import Image


def preprocess_image(image, target_size=(28, 28)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image


# Load the CoreML model
model = ct.models.MLModel('MNISTClassifier.mlmodel')

for digit in range(1,9):
    # Load the image
    digit_dir = 'dataset/' + str(digit)
    files = os.listdir(digit_dir)

    # Filter out files to ensure they are images
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(digit_dir, image_file)
        
        # Open and preprocess the image
        image = Image.open(image_path)

        # Convert the image to a numpy array
        image_data = np.array(image).astype(np.float32)

        # Normalize the image
        image_data = image_data / 255

        # Add a batch dimension
        image_data = image_data[np.newaxis, :, :]

        # Make a prediction
        prediction = model.predict({'image': image_data})

        # Print the prediction
        print(f'The digit is: {digit}')
        print(f'The model predicted: {prediction["classLabel"]}')
        print(f'The model\'s confidence: {prediction["classLabelProbs"][prediction["classLabel"]]}')
        print()

# # Load and preprocess images
# image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']
# images = [Image.open(img_path) for img_path in image_paths]

# preprocessed_images = [preprocess_image(img) for img in images]

# # Run inference
# predictions = [model.predict({'image': img}) for img in preprocessed_images]

# # Create a Turi Create SFrame from predictions
# data = {
#     'image_path': image_paths,
#     'prediction': [pred['classLabel'] for pred in predictions]
# }

# sf = tc.SFrame(data)

# # Optionally, add more columns to the SFrame
# sf['image'] = tc.Image(image_paths)

# # Save the SFrame
# sf.save('output.sframe')