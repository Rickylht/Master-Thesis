from bisect import bisect_left
import random
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shutil
import time

FUSIONPATH1 = '.\\fusion_workplace\\img1.bmp'
FUSIONPATH2 = '.\\fusion_workplace\\img2.bmp'
FUSIONRESULT_PATH = '.\\fusion_workplace\\fusion.bmp'

def fusion_sift(imgpath1, imgpath2):

    MIN_MATCH_COUNT = 3

    img1 = cv2.imread(imgpath1,0) # queryImage
    img2 = cv2.imread(imgpath2,0) # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        matchesMask = mask.ravel().tolist() 
        print(M)
        print(matchesMask)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2) 

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray'),plt.show()

        #draw image frame
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        _img1 = cv2.warpPerspective(img1,M, (w,h))
        added_image = cv2.addWeighted(img2, 0.5, _img1, 0.5, 0)
        plt.imshow(added_image, 'gray'),plt.show()

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        return

def fusion_contour(img1, img2):
    '''
        use template matching to find best overlay and do image fusion
    '''

    assert img1.shape == img2.shape
    cv2.imwrite(FUSIONPATH1, img1)
    cv2.imwrite(FUSIONPATH2, img2)
    hmerge = np.hstack((img1 , img2))
    plt.imshow(hmerge, 'gray'),plt.show()

    ret, thresh_1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY)
    ret, thresh_2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY)
    cv2.imwrite('.\\fusion_workplace\\thresh_1.bmp', thresh_1)
    cv2.imwrite('.\\fusion_workplace\\thresh_2.bmp', thresh_2)
    thresh_merge = np.hstack((thresh_1, thresh_2))
    plt.imshow(thresh_merge, 'gray'),plt.show()

    time_start = time.time()

    contours_1, hierarchy_1 = cv2.findContours(thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = cv2.boundingRect(contours_1[0])
    #make a bigger square template so that no distortion after rotation
    template = thresh_1[int(y+h/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(y+h/2+pow((pow(h/2,2) + pow(w/2,2)),0.5)), 
                        int(x+w/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(x+w/2+pow((pow(h/2,2) + pow(w/2,2)),0.5))] # as template
    
    best_pro = 1.0
    best_degree = 0
    best_loc = (0 ,0)
    best_scale = 1.0

    (h, w) = template.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    #rotate between -15° to 15° step 0.5° , scaling from 0.9 to 1.1 step 0.01
    for scale in np.linspace(0.9, 1.1, num = 21):
        for degree in np.linspace(-15, 15, num = 61): 
            dim = (int(w * scale), int(h * scale))
            resized = cv2.resize(template, dim, interpolation = cv2.INTER_AREA)
            M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
            rotated = cv2.warpAffine(resized, M, (w, h))
            res = cv2.matchTemplate(thresh_2, rotated, 1)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val <= best_pro:
                best_pro = min_val
                best_degree = degree
                best_loc = min_loc
                best_scale = scale
    
    print("best probability = ",  best_pro)
    print("best degree = ", best_degree)
    print("best location = ", best_loc)
    print("best scale = ", best_scale)
    
    #put mask on image1
    x,y,w,h = cv2.boundingRect(contours_1[0])
    masked_1 = img1[int(y+h/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(y+h/2+pow((pow(h/2,2) + pow(w/2,2)),0.5)), 
                    int(x+w/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(x+w/2+pow((pow(h/2,2) + pow(w/2,2)),0.5))]
    (h, w) = masked_1.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    best_dim = (int(w * best_scale), int(h * best_scale))
    resized = cv2.resize(masked_1, best_dim, interpolation = cv2.INTER_AREA)
    M = cv2.getRotationMatrix2D((cX, cY), best_degree, 1.0)
    best_template = cv2.warpAffine(resized, M, (w, h))
    #plt.imshow(best_template, 'gray'),plt.show()
    

    def overlay(largeImg, smallImg, regionTopLeftPos = (0,0)):
        srcW, srcH = largeImg.shape[1::-1]
        refW, refH = smallImg.shape[1::-1]
        x,y =  regionTopLeftPos
        if (refW > srcW) or (refH > srcH):
            print("image size error")
            return
        else:
            if (x + refW) > srcW:
                x = srcW - refW
            if (y + refH)> srcH:
                y = srcH - refH
            
            destImg = np.array(largeImg)
            tmpSrcImg = destImg[y:y+refH,x:x+refW]

            tmpImg = np.zeros((refH, refW))
            for i in range(refH):
                for j in range(refW):
                    tmpImg[i,j] = min(smallImg[i,j], tmpSrcImg[i,j])
                              
            destImg[y:y + refH, x:x + refW] = tmpImg
  
            return destImg
    
    img3 = overlay(img2, best_template, (best_loc))
    cv2.imwrite(FUSIONRESULT_PATH, img3)

    time_end = time.time()
    print('Time cost: ',time_end - time_start,'s')

    plt.imshow(img3, 'gray'),plt.show()

    return img3

def two_image_fusion(imgpath1, imgpath2):

    img1 = cv2.imread(imgpath1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_GRAYSCALE)
    fusion_contour(img1, img2)

def four_image_fusion(imgpath_0, imgpath_30, imgpath_60, imgpath_90):
    
    img0 = cv2.imread(imgpath_0, cv2.IMREAD_GRAYSCALE)
    img30 = cv2.imread(imgpath_30, cv2.IMREAD_GRAYSCALE)
    img60 = cv2.imread(imgpath_60, cv2.IMREAD_GRAYSCALE)
    img90 = cv2.imread(imgpath_90, cv2.IMREAD_GRAYSCALE)

    assert img0.shape == img30.shape == img60.shape == img90.shape

    (h, w) = img0.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M_30 = cv2.getRotationMatrix2D((cX, cY), -30, 1.0)
    img30 = cv2.warpAffine(img30, M_30, (w, h))

    M_60 = cv2.getRotationMatrix2D((cX, cY), -60, 1.0)
    img60 = cv2.warpAffine(img60, M_60, (w, h))

    M_90 = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    img90 = cv2.warpAffine(img90, M_90, (w, h))

    tmp_img1 = fusion_contour(img0, img60)
    tmp_img2 = fusion_contour(img30, img90)
    tmp_img3 = fusion_contour(tmp_img1, tmp_img2)

    cv2.imwrite(FUSIONRESULT_PATH, tmp_img3)
    

if __name__ == '__main__':
    #fusion_sift(".\\data\\masked\\010_830_h.bmp", ".\\data\\masked\\010_830_v.bmp")
    
    two_image_fusion(".\\fusion_workplace\\source\\masked\\002_h.bmp", ".\\fusion_workplace\\source\\masked\\002_v.bmp")

    #four_image_fusion(".\\fusion_workplace\\source\\masked\\005_0.bmp", ".\\fusion_workplace\\source\\masked\\005_30.bmp", ".\\fusion_workplace\\source\\masked\\005_60.bmp", ".\\fusion_workplace\\source\\masked\\005_90.bmp")

    pass
