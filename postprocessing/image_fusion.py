from bisect import bisect_left
import random
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import shutil
 
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

def fusion_contour(imgpath1, imgpath2):

    img1 = cv2.imread(imgpath1,0) 
    img2 = cv2.imread(imgpath2,0) 

    shutil.copy(imgpath1, '.\\fusion_workplace\\img1.bmp')
    shutil.copy(imgpath2, '.\\fusion_workplace\\img2.bmp')
    hmerge = np.hstack((img1 , img2))
    plt.imshow(hmerge, 'gray'),plt.show()

    ret, thresh_1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY)
    ret, thresh_2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY)
    cv2.imwrite('.\\fusion_workplace\\thresh_1.bmp', thresh_1)
    cv2.imwrite('.\\fusion_workplace\\thresh_2.bmp', thresh_2)
    thresh_merge = np.hstack((thresh_1, thresh_2))
    plt.imshow(thresh_merge, 'gray'),plt.show()

    contours_1, hierarchy_1 = cv2.findContours(thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = cv2.boundingRect(contours_1[0])
    template = thresh_1[int(y+h/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(y+h/2+pow((pow(h/2,2) + pow(w/2,2)),0.5)), 
                        int(x+w/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(x+w/2+pow((pow(h/2,2) + pow(w/2,2)),0.5))] # as template
    
    #plt.imshow(template, 'gray'),plt.show()
    #plt.imshow(thresh1, 'gray'),plt.show()

    best_pro = 1
    best_degree = 0
    best_loc = (0,0)

    (h, w) = template.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    #rotate between -30° to 30°
    for i in range(60):    
        degree = i - 30    
        M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
        rotated = cv2.warpAffine(template, M, (w, h))
        res = cv2.matchTemplate(thresh_2, rotated, 1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if min_val <= best_pro:
            best_pro = min_val
            best_degree = degree
            best_loc = min_loc
    
    print("best_pro = ",  best_pro)
    print("best degree = ", best_degree)
    print("best location = ", best_loc)

    #fusion
    x,y,w,h = cv2.boundingRect(contours_1[0])
    masked_1 = img1[int(y+h/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(y+h/2+pow((pow(h/2,2) + pow(w/2,2)),0.5)), 
                    int(x+w/2-pow((pow(h/2,2) + pow(w/2,2)),0.5)):int(x+w/2+pow((pow(h/2,2) + pow(w/2,2)),0.5))]
    # plt.imshow(masked_1, 'gray'),plt.show()
    (h, w) = masked_1.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), best_degree, 1.0)
    rotated = cv2.warpAffine(masked_1, M, (w, h))
    #plt.imshow(rotated, 'gray'),plt.show()

    def addWeightedSmallImgToLargeImg(largeImg,alpha,smallImg,beta,gamma=0.0,regionTopLeftPos=(0,0)):
        srcW, srcH = largeImg.shape[1::-1]
        refW, refH = smallImg.shape[1::-1]
        x,y =  regionTopLeftPos
        if (refW>srcW) or (refH>srcH):
            #raise ValueError("img2's size must less than or equal to img1")
            raise ValueError(f"img2's size {smallImg.shape[1::-1]} must less than or equal to img1's size {largeImg.shape[1::-1]}")
        else:
            if (x+refW)>srcW:
                x = srcW-refW
            if (y+refH)>srcH:
                y = srcH-refH
            destImg = np.array(largeImg)
            tmpSrcImg = destImg[y:y+refH,x:x+refW]
            #print("tmpSrcImg shape = ", tmpSrcImg.shape)

            tmpImg = np.zeros((refH, refW))
            for y in range(refH):
                for x in range(refW):
                    pixel_small = smallImg[y,x]
                    pixel_tmpSrc = tmpSrcImg[y,x]
                    if pixel_small < pixel_tmpSrc:
                        tmpImg[y,x] = pixel_small
                    else:
                        tmpImg[y,x] = pixel_tmpSrc
            #plt.imshow(tmpImg, 'gray'),plt.show()
            
            #tmpImg = cv2.addWeighted(tmpSrcImg, alpha, smallImg, beta,gamma)
            #print("tmpImg shape = ", tmpImg.shape)
            #destImg[y:y + refH, x:x + refW] = tmpImg
            return tmpImg
    
    img3 = addWeightedSmallImgToLargeImg(img2, 0.5, rotated, 0.5, 0.0, (best_loc))
    cv2.imwrite('.\\fusion_workplace\\fusion.bmp', img3)
    plt.imshow(img3, 'gray'),plt.show()


if __name__ == '__main__':
    #fusion_sift(".\\data\\masked\\010_830_h.bmp", ".\\data\\masked\\010_830_v.bmp")
    fusion_contour(".\\data\\masked\\009_830_h.bmp", ".\\data\\masked\\009_830_v.bmp")
    pass 
