import cv2
import numpy as np
import json

def create_mask(imgpath, jsonpath):
    '''create single mask for one image and save'''
    with open(jsonpath, "r") as f:
        data = f.read()
    
    jsondata = json.loads(data)
    for entry in jsondata['shapes']:
        if entry['label'] == 'tooth':
            tooth_point_list = entry['points']
            tooth_point_list = np.array(tooth_point_list, dtype=np.int32)
            image = cv2.imread(imgpath)
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [tooth_point_list], (255, 255, 255))
            cv2.imwrite('.\data\masks\sample.png', mask)

#create_mask(".\data\imgs\sample.bmp", ".\data\jsons\sample.json")

def create_all_masks():
    '''create masks for all images and save'''
    pass

def visualize_masks(number):
    '''visualize number of masks'''
    pass

def prepocessing():
    '''image processing before training'''
    pass