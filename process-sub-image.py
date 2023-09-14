import cv2 as cv
import numpy as np
import random as rng
rng.seed(99999)

# read input image
THRESH_MINI = 150                            # pixel value above which you want MAX VALUE 
THRESH_MAX_VAL = 160                          # pixel value above our THRESH_MIN will be converted to this value 
MIN_SIZE = 800                            # minimum size of enclosed figures you want to use (function - noise_removal)

def noise_removal(image):
    image = image.astype( 'uint8' )
    # Connected components analysis
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    # min_size == Hyperparameter depends upon YOUR USAGE
    min_size = MIN_SIZE
    # Fake image where we put all the candidate figures
    # image bigger than our parameter are only accepted
    img2 = np.zeros(( output.shape ))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2.astype('uint8')

def filling(im_th):
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8) 
    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return [im_floodfill,im_floodfill_inv]

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def draw_bounding_boxes(threshold,given_img):

    # Standard Canny Edge Detection Implementation
    canny_output = cv.Canny(given_img, threshold, threshold * 2)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))


    bigg=boundRect[0]
    for rec in boundRect:
        if ((bigg[2]*bigg[3]) < (rec[2]*rec[3])):
            bigg = rec
    print(bigg)
    cv.rectangle(drawing, (int(bigg[0]), int(bigg[1])), (int(bigg[0]+bigg[2]), int(bigg[1]+bigg[3])), color, 2)
    crop = src_gray[int(bigg[1]):int(bigg[1])+int(bigg[3]) , int(bigg[0]):int(bigg[0]+bigg[2])]
    # drawing the rectangles on extracted coordinates
    #for i in range(len(contours)):
        #cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        ## For cropping the blocks
        # crop = src_gray[int(boundRect[i][1]):int(boundRect[i][1]+boundRect[i][3]) , int(boundRect[i][0]):int(boundRect[i][0]+boundRect[i][2]) ]
        # cv.imshow("contour"+str(i),crop)
    cv.imshow('Contours', drawing)
    cv.imshow('crop', crop)


src  = cv.imread('subimage/img_5.jpg',cv.IMREAD_GRAYSCALE)

# Binarization modifications
src_gray = cv.blur(src, (3,3))
th, im_th = cv.threshold(src_gray, THRESH_MINI , THRESH_MAX_VAL, cv.THRESH_BINARY_INV)
#cv.imshow('Character im_th' , im_th)
im_th = noise_removal(im_th)
#cv.imshow('Character Removal' , im_th)




kernel = np.ones((5, 5), np.uint8)
img_dilation = cv.dilate(im_th, kernel, iterations=2)
cv.imshow("Inverted Floodfilled Image", img_dilation)

# #filling
# im_floodfill,im_floodfill_inv = filling(im_th)
# cv.imshow("Floodfilled Image", im_floodfill)
# cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)

draw_bounding_boxes(700,im_th)

cv.waitKey()