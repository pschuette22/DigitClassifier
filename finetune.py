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
model = ct.models.MLModel('product/MNISTClassifier.mlpackage')

print('Processing dataset')
for digit in range(1,10):
    print(f'Processing {digit}')
    # Load the image
    digit_dir = 'dataset/' + str(digit)
    files = os.listdir(digit_dir)
    print('Image directory: ' + digit_dir)
    # Filter out files to ensure they are images
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(digit_dir, image_file)
        
        # Open and preprocess the image
        image = Image.open(image_path).convert('L')

        # Convert the image to a numpy array
        image_data = np.array(image).astype(np.float32)

        # Normalize the image
        image_data = image_data / 255

        # # Add a batch dimension
        # image_data = image_data[np.newaxis, :, :]

        # Make a prediction
        prediction = model.predict({'image': image})

        # Print the prediction
        print('The digit is: ', digit)
        print('The model predicted: ', prediction["classLabel"])
        # print(repr(prediction))
        print('The model\'s confidence: ', prediction["labelProbabilities"][prediction["classLabel"]])
        if digit != prediction["classLabel"]:
            print("Path: ", image_path)
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