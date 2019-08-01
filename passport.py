import cv2
import re
import imutils
from imutils.object_detection import non_max_suppression
from imutils.contours import sort_contours
import numpy as np
from pytesseract import image_to_string
from PIL import Image
from points_transform import four_point_transform


DEBUG = False
DEBUG_MRZ_ZONE = False
SHOW_CUT = False
DEBUG_TEXT_SKEW_CORRECTION = False
SHOW_MRZ_CNTRS = False
SHOW_CUT_THRESH = False
SHOW_CUT_MASK = False
SHOW_RAW_MRZ = False
SHOW_SKEW_CORRECTION_ATTEMPTS = False
trained_data_config = r'--tessdata-dir "./tessdata"'


def skew_text_correction(image, type_=1):
    ratio = image.shape[0] / 500.0
    
    orig = image.copy()
    
    if image.shape[1] > 1000:
        image = imutils.resize(image, width=1000)
    image = imutils.resize(image, height = 500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if type_ == 1:
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(gray, 10, 300)
        
    elif type_ == 2:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        edged = cv2.Canny(gray, 0, 200)

    elif type_ == 3:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        edged = cv2.Canny(gray, 10, 100)

    elif type_ == 4 or type_ == 5 or type_ == 7:
        gray = cv2.GaussianBlur(gray, (5,3), 0)
        edged = cv2.Canny(gray, 10, 80)
        
    elif type_ == 6:
        gray = cv2.GaussianBlur(gray, (9,9), 0)
        edged = cv2.Canny(gray, 5, 50)
 
    elif type_ == 8:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        edged = cv2.Canny(gray, 0, 40)

    if type_ == 9:
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(gray, 10, 400)
        
    if type_ == 10:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        edged = cv2.Canny(gray, 90, 250)
        
    if DEBUG_TEXT_SKEW_CORRECTION:
        cv2.imshow('win', imutils.resize(edged.copy(), width=500))
        cv2.waitKey(0)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    dilated = cv2.dilate(edged,kernel,iterations = 1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,10))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    ret, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    max_peri = 0
    img_area = thresh.shape[0]*thresh.shape[1]
    area_thresh = 5000
    screenCnt = None
    for c in cnts:
    	area = cv2.contourArea(c)
    	peri = cv2.arcLength(c,True)
    	approx = cv2.approxPolyDP(c, 0.07*peri, True)
    	if type_ == 5:
    	    approx = cv2.approxPolyDP(c, 0.001*peri, True)
    	elif type_ == 7:
    	    approx = cv2.approxPolyDP(c, 0.09*peri, True)
    	elif type_ == 10:
    	    approx = cv2.approxPolyDP(c, 0.07*peri, True)
    	cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    	if DEBUG_TEXT_SKEW_CORRECTION:
    	    print(len(approx))
    	if len(approx) == 4:
    	    print(img_area - area)
    	    if peri > max_peri and img_area - area > area_thresh:
    		    max_peri = peri
    		    screenCnt = approx
                
    if DEBUG_TEXT_SKEW_CORRECTION:
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 1)
        cv2.imshow("win", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if screenCnt is None:
        return None
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    print("passed")
    return warped


def passport_border(image):
    '''
    Убирает границы паспорта
    Параметры:
    image - картинка (numpy array)
    Возвращает картинку (numpy array)
    '''
    
    #image = cv2.imread(filename)
    #image = imutils.resize(image, width=1000)
    cascade = cv2.CascadeClassifier('cascade.xml')
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # находим лицо
    face = cascade.detectMultiScale(gray, 1.3, 5)

    if face is not ():
        (x, y, w, h) = face[0]

        (H, W, _) = image.shape

        if y + 3 * h > H:
            endY = H
        else:
            endY = y + 3 * h

        if x - w < 0:
            startX = 0
        else:
            startX = x - w

        startY = 0
        endX = W

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[startY:endY, startX:endX] = 255

        masked = cv2.bitwise_and(image, image, mask=mask)
        masked = get_segment_crop(image, mask=mask)

        return masked

    else:
        return image


def rotate_passport(passport):
    """
    Поворачивает картинку в вертикальное положение
    Параметры:
    passport - картинка (numpy array)
    Возвращает повернутую картинку (numpy array) или
    None, если картинка не является паспортом (если не найдено лицо)
    """

    cascade = cv2.CascadeClassifier('cascade.xml')
    image = imutils.resize(passport.copy(), width=1000)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        return None

    rotates = 0
    for _ in range(4):
        # поворачиваем и ищем лицо, если находим - возвращаем картинку
        face = cascade.detectMultiScale(gray, 1.3, 5)

        if face is not ():
            return imutils.rotate_bound(passport, 90 * rotates)

        gray = imutils.rotate_bound(gray, 90)
        rotates += 1

    return None


def get_segment_crop(img,tol=0, mask=None, border=False):
    """
    Возвращает часть картинки (numpy array), вырезанную по маске
    Параметры:
    img - картинка (numpy array)
    tol - если маска не указана, то для вырезания используется данный
    параметр. возвращается часть картинки, в которой пиксели имеют
    значение больше tol
    mask - черно-белая картинка (numpy array) с маской
    имеет такие же размеры, как исходная картинка. вырезается часть
    картинки, соответствующая белой части маски
    """
    
    if mask is None:
        mask = img > tol
        
    try:
        if border:
            ix = np.ix_(mask.any(1), mask.any(0))
            
            edge = 40
            
            ix0 = np.arange(ix[0][0][0]-edge, ix[0][-1][0]+edge)
            ix1 = np.arange(ix[1][0][0]-edge, ix[1][0][-1]+edge)

            ix0 = ix0[(ix0[:]>0) & (ix0[:]< img.shape[0])]
            ix1 = ix1[(ix1[:]>0) & (ix1[:]<img.shape[1])]

            ix0 = ix0.reshape(-1,1)
            ix1 = ix1.reshape(1,-1)
            
            return img[[ix0, ix1]]
        else:
            return img[np.ix_(mask.any(1), mask.any(0))]
    except:
        raise
    
def cut_passport(image, type_='passport'):
    """
    Вырезает паспорт из картинки
    Параметры:
    image - картинка (numpy array)
    Возвращает картинку (numpy array) с вырезанным паспортом
    """
    
    image = image.copy()
    # image = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    blended = cv2.addWeighted(src1=sobelX, alpha=0.5, src2=sobelY, beta=0.5, gamma=0)

    kernel = np.ones((20, 20), dtype=np.uint8)
    opening = cv2.morphologyEx(blended, cv2.MORPH_OPEN, kernel)

    min_ = np.min(opening)
    opening = opening - min_
    max_ = np.max(opening)
    div = max_/255
    opening = np.uint8(opening / div)

    blurred = cv2.GaussianBlur(opening, (1, 1), 0)
    thresh = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    kernel = np.ones((2,2), dtype=np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    if SHOW_CUT_THRESH:
        cv2.imshow('win', imutils.resize(thresh.copy(), width=1000))
        cv2.waitKey(0)
    
    (h, w) = thresh.shape
    if type_=='passport':
        edgeH = int(h * 0.01)
        edgeW = int(w * 0.01)
        thresh[0:edgeH,0:w] = 255
        thresh[h-edgeH:h,0:w] = 255
        thresh[0:h,0:edgeW] = 255
        thresh[0:h, w-edgeW:w] = 255


    kernel = np.ones((20, 20), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    inverse = cv2.bitwise_not(thresh)
    
    if SHOW_CUT_MASK and type_=='mrz':
        cv2.imshow('win', imutils.resize(inverse.copy(), width=1000))
        cv2.waitKey(0)

    if type_=='mrz':
        masked = get_segment_crop(image, mask=inverse, border=True)
    else:
        masked = get_segment_crop(image, mask=inverse)

    return masked


def authority_text_boxes(image, boxes, rW, rH):
    '''
    Возвращает список ROI (картинки в виде numpy массивов) в верхней части паспорта
    Параметры:
    image - картинка (numpy array)
    boxes - 
    rW, rH - отношение ширины и высоты к 320 (возвращаемое функцией locate_text)
    '''
    
    mask = np.zeros(image.shape[:2],dtype=np.uint8)
    mask_text_zones = np.zeros(image.shape[:2],dtype=np.uint8)
    (H,W) = image.shape[:2]

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        if startX > 0.2 * W and endX < 0.8 * W:

            mask[startY:endY,:] = 255
            mask_text_zones[startY:endY,startX:endX] = 255

    _, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    temp_masks = []
    authority = []
    
    try:
        (cnts, boundingBoxes) = sort_contours(cnts, method='top-to-bottom')
    except:
        return []
    
    for k, (cnt, box)in enumerate(zip(cnts, boundingBoxes)):

        temp_mask = np.zeros(mask.shape[:2],dtype=np.uint8)
        temp_mask = cv2.drawContours(temp_mask, [cnt], -1, (255,255,255), -1)

        temp_masks.append(temp_mask)

        line_test = cv2.bitwise_and(temp_mask, mask_text_zones)

        (H,W) = image.shape[:2]

        left_border = np.where(line_test == 255)[1].min()
        right_border = np.where(line_test == 255)[1].max()
        top_border = np.where(line_test == 255)[0].min()
        bottom_border = np.where(line_test == 255)[0].max()

        """left_border = max(left_border - 250, int(0.2 * W))
        right_border = min(right_border + 250, int(0.8 * W))"""

        left_border = 0
        right_border = W
        top_border = max(top_border - 20, 0)
        bottom_border = min(bottom_border + 20, H)

        line_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        line_mask[top_border:bottom_border, left_border:right_border] = 255

        masked_line = get_segment_crop(image, image, mask=line_mask)
        authority.append(masked_line)

    # ROIs, mask, mask_text_zones
    return authority


def name_text_boxes(image, boxes, rW, rH):
    '''
    Возвращает список ROI (картинки в виде numpy массивов) в нижней части паспорта
    Параметры:
    image - картинка (numpy array)
    boxes - список из numpy массивов с 4-мя координатами ROI (возвращаемых функцией locate_text)
    rW, rH - отношение ширины и высоты к 320 (возвращаемое функцией locate_text)
    '''
    
    mask = np.zeros(image.shape[:2],dtype=np.uint8)
    mask_text_zones = np.zeros(image.shape[:2],dtype=np.uint8)
    (H,W) = image.shape[:2]
    
    if DEBUG:
        cv2.imshow('win', image)
        cv2.waitKey(0)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        if startX >= 0.1 * W and endX <= 0.8 * W:

            mask[startY:endY,:] = 255
            mask_text_zones[startY:endY,startX:endX] = 255

    _, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    temp_masks = []
    text_boxes = []
    
    if not cnts:
        print("No contours found")
        return None
    
    (cnts, boundingBoxes) = sort_contours(cnts, method='top-to-bottom')
    for k, (cnt, box)in enumerate(zip(cnts, boundingBoxes)):

        temp_mask = np.zeros(mask.shape[:2],dtype=np.uint8)
        temp_mask = cv2.drawContours(temp_mask, [cnt], -1, (255,255,255), -1)

        temp_masks.append(temp_mask)

        line_test = cv2.bitwise_and(temp_mask, mask_text_zones)

        (H,W) = image.shape[:2]

        if k < 3:

            left_border = np.where(line_test == 255)[1].min()
            right_border = np.where(line_test == 255)[1].max()
            top_border = np.where(line_test == 255)[0].min()
            bottom_border = np.where(line_test == 255)[0].max()

            left_border = max(left_border - 150, 0)
            right_border = min(right_border + 150, W)
            top_border = max(top_border - 15, 0)
            bottom_border = min(bottom_border + 15, H)

        else:

            top_border = np.where(line_test == 255)[0].min()
            bottom_border = np.where(line_test == 255)[0].max()

            left_border = max(left_border - 150, 0)
            right_border = min(right_border + 150, W)
            top_border = max(top_border - 15, 0)
            bottom_border = min(bottom_border + 15, H)


        line_mask = np.zeros(temp_mask.shape[:2], dtype=np.uint8)
        line_mask[top_border:bottom_border, left_border:right_border] = 255

        masked_line = get_segment_crop(image, image, mask=line_mask)
        text_boxes.append(masked_line)

    # ROIs, mask, mask_text_zones
    return text_boxes


def locate_text(image):
    '''
    Ищет координаты ROI с текстом
    Параметры:
    image - картинка (numpy array)
    Возвращает список координат ROI (numpy массивы из 4-х значений)
    и кортеж (rW, rH) (отношение ширины и высоты к 320) или None
    '''
    
    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    net = cv2.dnn.readNet('EAST.pb')

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    try:
        (scores, geometry) = net.forward(layerNames)
    except:
        return None

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.01:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    boxes = sorted(boxes,key=lambda x:x[1])

    return boxes, (rW, rH)


def read_text(roi, type_):    
    '''
    Распознает текст с картинки
    Параметры:
    roi - картинка (numpy array)
    type_ - строка типа текста (auth, birth, number, name)
    тип определяет regex фильтры, применяемые к тексту
    Возвращает текст
    '''
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    eroded = cv2.erode(blurred, (3,3), iterations=1)
    dilated = cv2.dilate(eroded, (3,3), iterations=1)
    ret, thresh = cv2.threshold(dilated,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = image_to_string(thresh, lang='rus').replace('\n', ' ')
    
    if DEBUG:
        cv2.imshow('win', thresh)
        cv2.waitKey(0)
        print(text)
    
    if type_ == 'auth':

        text = re.sub(r'[^А-Я\.\d -]+', '', text)

        ALLOWED_SHORT_WORDS = ['и', 'в', 'по']
        authority = ''
        for word in text.split():
            if len(word) > 2 or word.lower() in ALLOWED_SHORT_WORDS:
                authority += word + ' '

        text = authority

    elif type_ == 'birth':

        text = re.sub(r'[^А-Я\. -]+', '', text)

        ALLOWED_SHORT_WORDS = ['с.']
        birth_place = ''
        for word in text.split():
            if len(word) > 2 or word.lower() in ALLOWED_SHORT_WORDS:
                birth_place += word + ' '

        text = birth_place

    elif type_ == 'number':
        text = re.sub(r'[^\d. -]+', '', text)

    elif type_ == 'name':

        for word in text.split():

            word = re.sub(r'[^а-яА-Я ]+', '', word)
            potentials = word.split()
            if potentials != []:
                text = sorted(potentials, key=len)[-1]

    return text


def read_side(image):
    '''
    Распознает и считывает боковую сторону паспорта 
    Параметры:
    image - картинка (numpy array)
    Возвращает словарь с series и number (серия и номер)
    '''
    
    image = image.copy()
    image = imutils.rotate_bound(image, angle=-90)

    (h,w) = image.shape[:2]

    side = image[:h//5,:]

    gray = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    eroded = cv2.erode(blurred, (3,3), iterations=1)
    dilated = cv2.dilate(eroded, (3,3), iterations=1)
    ret, thresh = cv2.threshold(dilated,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = image_to_string(thresh)

    output = {'series': '', 'number': ''}

    series = re.search(r'\d{2} \d{2}', text)
    if series is not None:
        output['series'] = series[0]

    number = re.search(r'\d{6}', text)
    if number is not None:
        output['number'] = number[0]

    return output


def locate_MRZ(image):
    '''
    Ищет MRZ (machine readable zone)
    Параметры:
    image - картинка (numpy array)
    Возвращает вырезанную из картинки область MRZ
    '''
    
    image = image.copy()
    (H,W) = image.shape[:2]
    orig = image.copy()

    image = imutils.resize(image, height=600)
    (h,w) = image.shape[:2]
    
    kX = W / w
    kY = H / h

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    thresh = cv2.erode(thresh, None, iterations=3)

    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    if DEBUG_MRZ_ZONE:
        cv2.imshow('win', thresh)
        cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    image_copy = image.copy()
    
    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])
        
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255,0,0), 2)
        if ar > 5 and crWidth > 0.75:
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)

            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))
            
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0,255,0), 2)
            roi = image[y:y + h, x:x + w].copy()
            roi = orig[int(y*kY):int(y*kY + h*kY), int(x*kX):int((x+w)*kX)].copy()

            roi = cut_passport(roi, type_='mrz')
            return roi
    if SHOW_MRZ_CNTRS:
        cv2.imshow('win', image_copy)
        cv2.waitKey(0)
        cv2.imshow('win', roi)
        cv2.waitKey(0)

def parse_mrz(image):
    '''
    Парсит MRZ (machine readable zone)
    Параметры:
    image - картинка (numpy array)
    Возвращает словарь с полями паспорта, извлеченными из MRZ
    '''

    mrz = locate_MRZ(image)

    if mrz is None:
        return None
    
    try:
        gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
    except:
        return

    if gray.shape[1] > 1000:
        gray = imutils.resize(gray.copy(), width=1000)
    
    if DEBUG:
        cv2.imshow('win', gray)
        cv2.waitKey(0)
        print('read_text')

    text = image_to_string(gray, lang='ocrb', config=trained_data_config)

    # filter empty
    text = list(filter(lambda e: bool(e), text.split('\n')))
    
    if SHOW_RAW_MRZ:
        print('TEXT:')
        print(text)
        
    if len(text) >= 2:
        (top, bottom) = text[0], text[-1]
        top = top.replace(' ', '')
        bottom = bottom.replace(' ', '')
    else:
        return None
    
    words = [word for word in top.split('<') if len(word) >= 3]
    if len(words) >= 3:
        words = words[:3]
        words[0] = words[0][5:]


    translit = {"A":"А","B":"Б","V":"В","G":"Г","D":"Д","E":"Е","2":"Ё","J":"Ж",
                "Z":"З","I":"И","Q":"Й","K":"К","L":"Л","M":"М","N":"Н","O":"О",
                "P":"П","R":"Р","S":"С","T":"Т","U":"У","F":"Ф",
                "H":"Х","C":"Ц","3":"Ч","4":"Ш","W":"Щ","X":"Ъ","Y":"Ы",
                "9":"Ь","6":"Э","7":"Ю","8":"Я"}
    
    translit_keys = list(translit.keys())
    
    mrz_result = {'surname': '', 'name': '', 'patronymic': '', \
                'birth_date': '', 'issue_date': '', 'issue_code': '', 'number': '', 'series': '', 'sex': ''}

    for i, field in enumerate(['surname', 'name', 'patronymic'], 1):
        if len(words) >= i:
            mrz_result[field] = ''.join([translit[letter] if (letter in translit_keys) else letter for letter in words[i-1]])
            
    try:
        mrz_result['birth_date'] = '{}.{}.19{}'.format(bottom[13:19][4:6], bottom[13:19][2:4], bottom[13:19][0:2])
    except:
        mrz_result['birth_date'] = ''
    
    mrz_result['sex'] = 'male' if 'M' in bottom else 'female'
    
    try:
        mrz_result['issue_date'] = '{}.{}.20{}'.format(bottom[-15:-9][4:6], bottom[-15:-9][2:4], bottom[-15:-9][0:2])
    except:
        mrz_result['issue_date'] = ''
        
    try:
        mrz_result['issue_code'] = '{}-{}'.format(bottom[-9:-3][0:3], bottom[-9:-3][3:6])
    except:
        mrz_result['issue_code'] = ''

    try:
        mrz_result['series'] = bottom[:2]
    except:
        mrz_result['series'] = ''
    
    try:
        mrz_result['number'] = bottom[3:9]
    except:
        mrz_result['number'] = ''
        
    return mrz_result


def analyze_passport(passport):
    '''
    Анализирует паспорт
    Параметры:
    passport - картинка с паспортом
    Возвращает словарь с результатами распознавания
    '''
    
    image = passport.copy()
    image = rotate_passport(image)

    if image is None:
        return {'Error': 'Not a Passport'}
    
    image_cut = None
    for type_ in [2,3,1,4,5,6,7,8,9,10]:
        image_cut_ = skew_text_correction(image, type_=type_)
        if SHOW_SKEW_CORRECTION_ATTEMPTS:
            if image_cut is not None:
                cv2.imshow('win', imutils.resize(image_cut, width=500))
                cv2.waitKey(0)
        if image_cut is not None and image_cut.shape[0] > 1000 and \
        image_cut.shape[0] / image_cut.shape[1] > 1.3:
            print("!!!!!!!")
            print(image_cut.shape[0])
            print(image.shape[0])
            print(image_cut.shape[0] / image_cut.shape[1] > 1.3)
            image_cut = image_cut_
            break
    #image_cut=None
    
    if SHOW_CUT and image_cut is not None:
        cv2.imshow('win', imutils.resize(image_cut, width=500))
        cv2.waitKey(0)
        
    # если неправильно вырезали, то вырезаем другим методом
    if image_cut is None:
        print("Cut in another way")
        #image_cut = cut_passport(image, type_='mrz')
        image_cut = cut_passport(image)
        
        if SHOW_CUT:
            cv2.imshow('Output', imutils.resize(image_cut, width=500))
            cv2.waitKey(0)
     
    image = passport_border(image_cut)
    
    (h,w) = image.shape[:2]

    top = image[0:h//2,:]
    res = locate_text(top)
    if res is None:
        return {'Error': 'Error processing top of the passport'}
    boxes, (rW, rH) = res
    top = authority_text_boxes(top, boxes, rW, rH)

    bottom = image[h//2:h,w//3:w]
    res = locate_text(bottom)
    if res is None:
        return {'Error': 'Error processing bottom of the passport'}
    boxes, (rW, rH) = res
    bottom = name_text_boxes(bottom, boxes, rW, rH)
    
    image_cut = {'top': [], 'surname': np.ones((1,1), dtype=np.uint8), 'name': np.ones((1,1), dtype=np.uint8), \
                 'patronymic': np.ones((1,1), dtype=np.uint8), \
                 'birth_date': np.ones((1,1), dtype=np.uint8),'birth_place': []}

    ocr_result = {'top': '', 'surname': '', 'name': '', 'patronymic': '', 'birth_date': '','birth_place': '', \
                         'issue_date': '', 'issue_code': ''}

    for line in top:
        image_cut['top'].append(line)
        ocr_result['top'] += read_text(line, type_='auth') + ' '

    if ocr_result['top'] != '':

        issue_date = re.search(r'\d{2}\.\d{2}\.\d{4}', ocr_result['top'])
        if issue_date is not None:
            ocr_result['issue_date'] = issue_date[0]

        issue_code = re.search(r'\d{3}-\d{3}', ocr_result['top'])
        if issue_code is not None:
            ocr_result['issue_code'] = issue_code[0]

    if bottom is not None and len(bottom) > 3:

        image_cut['surname'] = bottom[0]
        image_cut['name'] = bottom[1]
        image_cut['patronymic'] = bottom[2]

        ocr_result['surname'] = read_text(bottom[0], type_='name').upper()
        ocr_result['name'] = read_text(bottom[1], type_='name').upper()
        ocr_result['patronymic'] = read_text(bottom[2], type_='name').upper()


    if bottom is not None and len(bottom) > 4:

        image_cut['birth_date'] = bottom[3]
        ocr_result['birth_date'] = read_text(bottom[3], type_='number')

        for line in bottom[3:]:
            image_cut['birth_place'].append(line)
            ocr_result['birth_place'] += read_text(line, type_='birth') + ' '

    ocr_result.update(read_side(image))

    result = {'ocr_result': ocr_result, 'cut': image_cut}
    
    mrz = parse_mrz(image)
    
    '''
    if mrz is not None:
        for key, value in mrz.items():
            result['ocr_result']['mrz_' + key] = value
    '''
    
    if mrz is not None:
        result['ocr_result']['mrz'] = mrz

    return result['ocr_result']
