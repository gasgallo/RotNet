"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./test dir
    - Loads the model
    - Invokes process_images function from evaluate.py with this image
    - Returns the output file generated at /output
"""

from __future__ import print_function

#import numpy as np
import os
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

#from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from evaluate import process_images

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
    
    model_location = load_model('/input/rotnet_resnet50.hdf5')
    process_images(model_location, input_filepath, output_filepath,
                   1, False)

    ######################

    return send_file(output_filepath, mimetype='image/jpg')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0')
