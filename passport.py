import cv2
import os
import re
import sys
import imutils
from imutils.object_detection import non_max_suppression
from imutils.contours import sort_contours
import numpy as np
from pytesseract import image_to_string


def passport_border(image, mode='reading'):

    # Initializing cascade
    #image = cv2.imread(filename)
    #image = imutils.resize(image, width=1000)
    cascade = cv2.CascadeClassifier('cascade.xml')
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Finding a face
    face = cascade.detectMultiScale(gray, 1.3, 5)

    if face is not ():
        # Cutting the image so only passport was left
        (x, y, w, h) = face[0]

        (H, W, _) = image.shape

        if mode == 'reading':
            # Mode designed to find passport border
            # as acurate as possible in order to read it

            if y - int(6 * h) < 0:
                startY = 0
            else:
                startY = y - int(6 * h)

            if y + 3 * h > H:
                endY = H
            else:
                endY = y + 3 * h

            if x - w < 0:
                startX = 0
            else:
                startX = x - w

            if x + 6 * w > W:
                endX = W
            else:
                endX = x + 6 * w

            mask = np.zeros((H, W), dtype=np.uint8)
            mask[startY:endY, startX:endX] = 255

            masked = cv2.bitwise_and(image, image, mask=mask)
            masked = get_segment_crop(image, mask=mask)

            """
                cv2.imwrite(os.path.join('output', output + '.png'), masked)

            else:
                cv2.imwrite(os.path.join('output', output + '.png'), image)
            """

            return masked

    else:
        return image


def rotate_passport(passport):
    """
    rotating an image so passport could be readed
    :param image: np array
    :return: np array
    """

    # Initializing cascade
    cascade = cv2.CascadeClassifier('cascade.xml')
    image = imutils.resize(passport.copy(), width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rotates = 0
    # Looking for a face
    for _ in range(4):

        face = cascade.detectMultiScale(gray, 1.3, 5)

        if face is not ():
            return imutils.rotate_bound(passport, 90 * rotates)

        gray = imutils.rotate_bound(gray, 90)
        rotates += 1

    (h, w, _) = passport.shape
    if w > h:
        passport = rotate_bound(passport,angle=-90)

    print('Falsed')
    # Return false if the given picture is not a passport
    return passport


def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def cut_passport(image):

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

    (h, w) = thresh.shape
    edgeH = int(h * 0.01)
    edgeW = int(w * 0.01)
    thresh[0:edgeH,0:w] = 255
    thresh[h-edgeH:h,0:w] = 255
    thresh[0:h,0:edgeW] = 255
    thresh[0:h, w-edgeW:w] = 255

    kernel = np.ones((20, 20), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    inverse = cv2.bitwise_not(thresh)

    """
    coords = np.column_stack(np.where(inverse > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    image = rotate_bound(image,angle=angle)
    inverse = rotate_bound(inverse,angle=angle)"""

    masked = get_segment_crop(image, mask=inverse)
    """
    if not os.path.exists('output1'):
        os.mkdir('output1')

    cv2.imwrite(os.path.join('output1', output + '.png'), masked)
    """
    return masked


def authority_text_boxes(image, boxes, rW, rH):

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

    img_cnt, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    temp_masks = []
    authority = []

    (cnts, boundingBoxes) = sort_contours(cnts, method='top-to-bottom')
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

        if startX >= 0.1 * W and endX <= 0.8 * W:

            mask[startY:endY,:] = 255
            mask_text_zones[startY:endY,startX:endX] = 255

    img_cnt, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    temp_masks = []
    text_boxes = []

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
    (scores, geometry) = net.forward(layerNames)

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

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    eroded = cv2.erode(blurred, (3,3), iterations=1)
    dilated = cv2.dilate(eroded, (3,3), iterations=1)
    ret, thresh = cv2.threshold(dilated,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = image_to_string(thresh, lang='rus').replace('\n', ' ')

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

    image = image.copy()
    image = imutils.rotate_bound(image, angle=-90)

    (h,w) = image.shape[:2]

    side = image[:h//10,:]

    gray = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    eroded = cv2.erode(blurred, (3,3), iterations=1)
    dilated = cv2.dilate(eroded, (3,3), iterations=1)
    ret, thresh = cv2.threshold(dilated,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = image_to_string(thresh, lang='rus').replace('\n', ' ')

    output = {'series': '', 'number': ''}

    series = re.search(r'\d{2} \d{2}', text)
    if series is not None:
        output['series'] = series[0]

    number = re.search(r'\d{6}', text)
    if number is not None:
        output['number'] = number[0]

    return output


def analyze_passport(passport):

    image = passport.copy()
    (h,w) = image.shape[:2]

    top = image[0:h//2,:]
    boxes, (rW, rH) = locate_text(top)
    top = authority_text_boxes(top, boxes, rW, rH)

    bottom = image[h//2:h,w//3:w]
    boxes, (rW, rH) = locate_text(bottom)
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


    if len(bottom) > 3:

        image_cut['surname'] = bottom[0]
        image_cut['name'] = bottom[1]
        image_cut['patronymic'] = bottom[2]

        ocr_result['surname'] = read_text(bottom[0], type_='name').upper()
        ocr_result['name'] = read_text(bottom[1], type_='name').upper()
        ocr_result['patronymic'] = read_text(bottom[2], type_='name').upper()


    if len(bottom) > 4:

        image_cut['birth_date'] = bottom[3]
        ocr_result['birth_date'] = read_text(bottom[3], type_='number')

        for line in bottom[4:]:
            image_cut['birth_place'].append(line)
            #print(read_text(line, type_='birth') + ' ')
            ocr_result['birth_place'] += read_text(line, type_='birth') + ' '

    ocr_result.update(read_side(image))

    result = {'ocr_result': ocr_result, 'cut': image_cut}

    return result['ocr_result']
