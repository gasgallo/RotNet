"""
Test script to check accuracy and debug what obtained with the
evaluate.py script
"""

# import the necessary packages
import argparse
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

# Utilities
from utils import generate_rotated_image

def test_RotNet(model, input_path):
    """
    Randomly rotates an image iterating by 0, 90, 180, 270 degrees and
    tests if the FaceRot model applys the right counter rotation.
    Finally prints the accuracy, number of not detected faces and number
    of corrupted files.
    """

    # Admitted input file extensions
    extensions = ['.jpg', '.jpeg', '.bmp', '.png']

    # Check if input is a single image or a directory
    if os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        image_paths = [os.path.join(input_path, f) for f in
                       os.listdir(input_path)
                       if os.path.splitext(f)[1].lower() in extensions]

    # Parameters
    accuracy = 0.0
    count = 0
    corr = 0
    rotations = []
    predicted_angles = []
    rotation_choice = [0, 90, 180, 270]
    input_shape = (224, 224, 3)

    for idx, image_path in enumerate(image_paths):
        print('no {}, path {}'.format(idx, secure_filename(image_path)))
        count += 1
        image = cv2.imread(image_path, 1)
        if image is None:
            corr += 1
            count -= 1
            print('Pic {} corrupted'.format(image_path))
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rotations.append(np.random.choice(rotation_choice))
        rotation_angle = rotations[idx]

        # generate the rotated image
        image = generate_rotated_image(image, rotation_angle, 
                                       size=input_shape[:2],
                                       crop_center=True,
                                       crop_largest_rect=True)

        # add dimension to account for the channels if the image is greyscale
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)

        # preprocess input images
        image = preprocess_input(np.expand_dims(image.astype('float32'), axis=0))
        
        predictions = model.predict(image)
        
        predicted_angles.append(np.argmax(predictions, axis=1)*90)
        if rotations[idx]==predicted_angles[idx]:
            accuracy += 1
            print('original {}, detected {} --> GOOD'.format(rotations[idx],predicted_angles[idx]))
        else:
            print('original {}, detected {} --> BAD'.format(rotations[idx],predicted_angles[idx]))

    return accuracy/count, len(image_paths), corr

if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='Path to model')
    ap.add_argument('input_path', help='path to input image')
    args = ap.parse_args()

    # initialize dlib's face detector (HOG+SVM-based)
    print('[INFO] Loading model...')
    model_location = load_model(args.model)

    # process the input and produce the output
    print('[INFO] Testing accuracy...')
    acc, tot, corr = test_RotNet(model_location,args.input_path)

    print('Total number of images tested: {}'.format(tot))
    print('The accuracy on detected faces is {}%'.format(acc*100))
    print('Corrupted pictures: {}'.format(corr))


			
