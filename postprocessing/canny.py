import cv2

# This is to get Canny edge detector results

img_original=cv2.imread('.\\data\\masked\\005_v.bmp')

cv2.namedWindow('Canny')

def nothing(x):
    pass

cv2.createTrackbar('threshold1','Canny',50,400,nothing)
cv2.createTrackbar('threshold2','Canny',100,400,nothing)

while(True):
   
    threshold1=cv2.getTrackbarPos('threshold1','Canny')
    threshold2=cv2.getTrackbarPos('threshold2','Canny')

    img_edges=cv2.Canny(img_original,threshold1,threshold2)

    cv2.imshow('original',img_original)
    cv2.imshow('Canny',img_edges)  
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
