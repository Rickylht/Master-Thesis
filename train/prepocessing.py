import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def show_img(img):
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_mask(imgpath, jsonpath, maskpath):
    '''
    create single mask for one image and save
        Arg:
            imgpath: path of image
            jsonpath: path of labelled json file
            maskpath: path of created mask to save
        Return: 
            None
    '''
    with open(jsonpath, "r") as f:
        data = f.read()
    jsondata = json.loads(data)
    for entry in jsondata['shapes']:
        if entry['label'] == 'tooth':
            tooth_point_list = entry['points']
            tooth_point_list = np.array(tooth_point_list, dtype=np.int32)
            image = cv2.imread(imgpath)
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [tooth_point_list], (255,255,255))
            cv2.imwrite(maskpath, mask)

#create_mask(".\data\imgs\sample01.bmp", ".\data\jsons\sample01.json", ".\data\masks\sample01.bmp")

def mask_on_image(imgsrcpath, maskpath, maskedImgpath):
    '''
    put single mask on single image
        Args:
            imgsrcpath: path of original img
            maskpath: path of mask
            maskedImgpath: path of created masked image to save
        Return:
            None
    '''
    srcImg = cv2.imread(imgsrcpath, cv2.IMREAD_GRAYSCALE)
    maskImg = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    maskedImg = cv2.bitwise_and(srcImg, srcImg, mask = maskImg)
    cv2.imwrite(maskedImgpath, maskedImg)

#mask_on_image(".\data\imgs\sample01.bmp", ".\data\masks\sample01.bmp", "masked_sample.bmp")

def create_all_masks():
    '''create masks for all images and save'''
    pass

def visualize_mask(maskpath):
    '''visualize a mask'''
    mask = cv2.imread(maskpath)
    show_img(mask)

def check_histogram(imgpath):
    '''check histogram of single image'''

    img = cv2.imread(imgpath)
    show_img(img)
    plt.hist(img.ravel(), 256)
    plt.show()

def prepocessing(rawpath, imgpath):
    '''single image processing before training
        Args:
            rawpath: path of raw image
            imgpath: path of image to save
    '''
    #change to grey sacle
    raw = cv2.imread(rawpath, cv2.IMREAD_GRAYSCALE)


    show_img(raw)
    cv2.imwrite(imgpath, raw)

#prepocessing("1.bmp", "2.bmp")