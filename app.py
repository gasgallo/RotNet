"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes ffwd_to_img function from evaluate.py with this image
    - Returns the output file generated at /output

Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: checkpoint
    - It is loaded from /input
"""

from __future__ import print_function

import cv2
import numpy as np
import os
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from utils import RotNetDataGenerator, crop_largest_rectangle, angle_error, rotate

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', '.bmp', '.png'])
app = Flask(__name__)


@app.route('/', methods=["POST"])
def correct_rotation():
    """
    Take the input image and rotate it if necessary
    """
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(filename):
        return BadRequest("Invalid file type")

    input_filepath = os.path.join('./test/', filename)
    output_filepath = os.path.join('./output/', filename)
    input_file.save(input_filepath)

    ######################
    
    print('Loading model...')
    model_location = load_model('/input/rotnet_resnet50.hdf5')

    print('Processsing input image(s)...')
    process_images(model_location, input_filepath, output_filepath,
                   1, False)

    ######################

    return send_file(output_filepath, mimetype='image/jpg')

def process_images(model, input_path, output_path,
                   batch_size=64, crop=True):
    extensions = ['.jpg', '.jpeg', '.bmp', '.png']

    output_is_image = False
    if os.path.isfile(input_path):
        image_paths = [input_path]
        if os.path.splitext(output_path)[1].lower() in extensions:
            output_is_image = True
            output_filename = output_path
            output_path = os.path.dirname(output_filename)
    else:
        image_paths = [os.path.join(input_path, f)
                       for f in os.listdir(input_path)
                       if os.path.splitext(f)[1].lower() in extensions]
        if os.path.splitext(output_path)[1].lower() in extensions:
            print('Output must be a directory!')

    predictions = model.predict_generator(
        RotNetDataGenerator(
            image_paths,
            input_shape=(224, 224, 3),
            batch_size=64,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        ),
        steps=len(image_paths)
    )

    predicted_angles = np.argmax(predictions, axis=1)

    if output_path == '':
        output_path = '.'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for path, predicted_angle in zip(image_paths, predicted_angles):
        image = cv2.imread(path)
        rotated_image = rotate(image, -predicted_angle*90)
        if crop:
            size = (image.shape[0], image.shape[1])
            rotated_image = crop_largest_rectangle(rotated_image, -predicted_angle, *size)
        if not output_is_image:
            output_filename = os.path.join(output_path, os.path.basename(path))
        cv2.imwrite(output_filename, rotated_image)



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0')
