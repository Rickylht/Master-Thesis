import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def mask_on_image(imgsrcpath, maskpath, maskedImgpath):

    '''put mask on image'''
    
    srcImg = cv2.imread(imgsrcpath, cv2.IMREAD_GRAYSCALE)
    maskImg = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    maskedImg = cv2.bitwise_and(srcImg, srcImg, mask = maskImg)
    cv2.imwrite(maskedImgpath, maskedImg)

def mask_threshold(maskpath = '.\\prediction_workplace\\prediction.bmp'):

    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape[0], mask.shape[1]
    etVal, threshold = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    eroded = cv2.erode(threshold, kernel)

    dilated = cv2.dilate(eroded, kernel)

    Gaussian = cv2.GaussianBlur(dilated, (5, 5), 0)

    contours, hierarchy = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = len(contours) > 0

    area = []

    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    black = np.zeros((h, w))
    mask = cv2.drawContours(black, contours, max_idx, 255, cv2.FILLED)

    cv2.namedWindow("mask",0)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

    cv2.imwrite('.\\prediction_workplace\\prediction_threshold.bmp', mask)
    mask_on_image('.\\prediction_workplace\\origin.bmp', '.\\prediction_workplace\\prediction_threshold.bmp', '.\\prediction_workplace\\masked.bmp')

    masked = cv2.imread('.\\prediction_workplace\\masked.bmp')
    cv2.imshow("mask", masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mask_threshold()