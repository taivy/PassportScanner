from PIL import Image
import numpy as np
from flask import render_template, Flask, request, make_response
import passport
import json


app = Flask(__name__)
DEBUG = True

# Для получения результатов распознавания отправить POST запрос на uploader
# с файлом картинки

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
    Обрабатывает картинку с паспортом
    Параметры:
    - image: картинка (numpy массив)
    Возвращает словарь с результатами распознавания
    """

    responce = passport.analyze_passport(image.copy())

    return responce


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
