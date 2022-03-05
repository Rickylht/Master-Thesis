import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def show_img(img):
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_path_list(path):
    list = []

    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            list.append(fullname)
    
    return list

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
    image = cv2.imread(imgpath)
    mask = np.zeros_like(image, dtype=np.uint8)
    for entry in jsondata['shapes']:
        if entry['label'] == 'tooth':
            tooth_point_list = entry['points']
            tooth_point_list = np.array(tooth_point_list, dtype=np.int32)
            cv2.fillPoly(mask, [tooth_point_list], (255,255,255))
        if entry['label'] == 'decay':
            decay_point_list = entry['points']
            decay_point_list = np.array(decay_point_list, dtype=np.int32)
            cv2.fillPoly(mask, [decay_point_list], (220,220,220))
        if entry['label'] == 'filling':
            filling_point_list = entry['points']
            filling_point_list = np.array(filling_point_list, dtype=np.int32)
            cv2.fillPoly(mask, [filling_point_list], (192,192,192))
        if entry['label'] == 'crack':
            crack_point_list = entry['points']
            #crack_point_list = np.array(crack_point_list, dtype=np.int32)
            cv2.line(mask, (int(crack_point_list[0][0]), int(crack_point_list[0][1])), (int(crack_point_list[1][0]), int(crack_point_list[1][1])), (64,64,64), 3)
        
    cv2.imwrite(maskpath, mask)
    #print('successful')


def creat_all_mask():
    '''create masks for all images and save'''
    img_list = get_path_list('.\\teeth_dataset\\image')
    json_list = get_path_list('.\\teeth_dataset\\json')

    assert len(img_list) == len(json_list)
    for i in range(len(img_list)):
        create_mask(img_list[i], json_list[i], img_list[i].replace('image', 'mask'))
    

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

def visualize_mask(maskpath):
    '''visualize a mask'''
    mask = cv2.imread(maskpath)
    show_img(mask)

def check_histogram(imgpath):
    '''check histogram of image'''
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


if __name__ == '__main__':
    #create_mask(".\\data\\imgs\\sample05.bmp", ".\\data\\jsons\\sample05.json", ".\\data\\masks\\sample05.bmp")
    #mask_on_image(".\\data\\imgs\\009_830_45.bmp", ".\\data\\masks\\009_830_45.bmp", ".\\data\\masked\\009_830_45.bmp")
    creat_all_mask()
    #check_histogram('prediction_image\masked.bmp')
    pass
    