import cv2
import numpy as np

# This is to change the image brightness and contrast

alpha = 0.3
beta = 80
img_path = ".\\data\\masked\\002_v.bmp"
img = cv2.imread(img_path)
img2 = cv2.imread(img_path)

def updateAlpha(x):
    global alpha, img, img2
    alpha = cv2.getTrackbarPos('Constrat', 'image')
    alpha = alpha * 0.01
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
def updateBeta(x):
    global beta, img, img2
    beta = cv2.getTrackbarPos('Lightness', 'image')
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))


cv2.namedWindow('image')
cv2.createTrackbar('Constrat', 'image', 0, 300, updateAlpha)
cv2.createTrackbar('Lightness', 'image', 0, 255, updateBeta)
cv2.setTrackbarPos('Constrat', 'image', 100)
cv2.setTrackbarPos('Lightness', 'image', 10)

while (True):

    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()