import numpy as np
import cv2
import os
import random

def get_path_list(path):
    list = []
    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            list.append(fullname)
    
    return list

def creat_dummy():
    
    #make dummy images
    image_path_list = get_path_list(".\\dummy2\\image\\horizontal\\830nm")
    
    for i in range(len(image_path_list)):

        path = image_path_list[i]
        print(path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(path.replace('image', 'mask'), cv2.IMREAD_GRAYSCALE)
        maskedImg = cv2.bitwise_and(image, image, mask = mask)
        center = (random.randint(550,750), random.randint(400,800))
        axeslength = (random.randint(30,100), random.randint(20,50))
        angle = random.randint(0,90)
        startAngle = 0
        endAngle = 360
        grey = random.randint(50,80)#grey scale from 50 to 100
        color = (grey, grey, grey)
        thickness = -1
        maskedImg = cv2.ellipse(maskedImg, center, axeslength, angle, startAngle, endAngle, color, thickness)
        #make new mask
        newmask = np.zeros_like(image)
        newmask = cv2.ellipse(newmask, center, axeslength, angle, startAngle, endAngle, (255,255,255), thickness)
        

        '''cv2.imshow("img", maskedImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow("img", newmask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break'''
        
        cv2.imwrite(path.replace('image', '_image'), maskedImg)
        cv2.imwrite(path.replace('image', '_mask'), newmask)
    
    print('creating dummy dataset finished')
    
if __name__ == '__main__':
    creat_dummy()
    pass