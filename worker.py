import cv2
import numpy as np
import os
import pytesseract
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

IMAGEPATH='im-updated.jpeg'
custom_config = r'--oem 3 --psm 6'


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def prepareForOCR(img):
    gray = get_grayscale(img)
    thresh = thresholding(gray)
    openingR = opening(thresh)
    cannyR = canny(gray)
    return cannyR


def start():
    # read input image
    img = cv2.imread(IMAGEPATH)

    # define border color
    lower = (0, 80, 110)
    upper = (0, 120, 150)

    # threshold on border color
    mask = cv2.inRange(img, lower, upper)

    # dilate threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # recolor border to white
    img[mask==255] = (255,255,255)

    # convert img to grayscale
    gray = get_grayscale(img)

    # otsu threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 

    # apply morphology open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = 255 - morph

    # find contours and bounding boxes
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]


    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > 2000:
            x,y,w,h = cv2.boundingRect(cntr)
            cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            bboxes.append((x,y,w,h))
            

    index=0
    for bbox in bboxes:
        (x,y,w,h) = bbox
        preparedImg=prepareForOCR(img.copy()[y-5:y+h+5,x-5:x+w+5])
        cv2.imwrite(os.path.join("subimage" , 'img_{}.jpg'.format(index)), preparedImg)
        index=index+1

    cv2.imshow("bboxes_img", bboxes_img)
    cv2.waitKey(0)


start()

        
    


