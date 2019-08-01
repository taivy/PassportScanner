from PIL import Image
import numpy as np
from flask import render_template, Flask, request
from pdf2image import convert_from_bytes
import passport
import json
import traceback

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

        f = request.files['file']
        
        try:
            if f.filename.endswith('pdf'):
                pages = convert_from_bytes(f.read())
                result = []
                for i, page in enumerate(pages):
                    print("Page: ", i)
                    try:
                        #image = recurse_extract(page)
                        image = np.array(page)
                    except:
                        traceback.print_exc()
                        continue
                    response = handle_passport(image)
                    result.append(response)
                    print(response)
            else:
                image = np.array(Image.open(f))
                response = handle_passport(image)
                print(response)
                result = [response]
            
        except:
            traceback.print_exc()
            result = []
            
        return json.dumps(result, ensure_ascii=False)


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
