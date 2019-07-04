from PIL import Image
import imutils
import cv2
import numpy as np
from flask import render_template, Flask, request, make_response
import passport
import json
import os
from pytesseract import image_to_string

app = Flask(__name__)
DEBUG = True


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        image = request.files['file']
        image = np.array(Image.open(image))

        responce = handle_passport(image)

        return json.dumps(responce, ensure_ascii=False)

def handle_passport(image):
    """
    Function for full handling of passport image
    :param image: np array
    :return: dict
    """

    responce = passport.analyze_passport(image.copy())

    return responce


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
