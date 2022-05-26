import os
import shutil
import torch
import cv2
from torchvision import transforms
from torchvision.utils import save_image
from unet import UNet
import numpy as np

TESTPATH = '.\\teeth_dataset\\image\\horizontal\\830nm\\gain_02\\003\\dry\\labial.bmp'

def test_single_image(path = TESTPATH):

    '''predict a single image'''

    shutil.copy(path, '.\\prediction_workplace\\origin.bmp')
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    net = UNet(n_channels=1, n_classes=1).cuda()
    
    # set 0 for contour segmentation, set 1 for caries estimation
    flag = 0

    if flag == 0:
        weights = 'best_seg.pth'
    else:
        weights = 'best_caries.pth'    

    if os.path.exists(weights):
            net.load_state_dict(torch.load(weights))
    else:
            print('weights not loaded')
            return
    
    h, w = image.shape[0], image.shape[1]

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((400, 500)), transforms.ToTensor()])

    net.eval()
    with torch.no_grad():
        input = transform(image).cuda()
        input = torch.unsqueeze(input, dim = 0)
        output = net(input)
        save_image(output, '.\\prediction_workplace\\prediction.bmp')

    mask = cv2.imread('.\\prediction_workplace\\prediction.bmp')
    mask = cv2.resize(mask,(w, h))
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('.\\prediction_workplace\\prediction.bmp', mask)
    
    print("prediction succesful")
    
    return


IMGSRCPATH = '.\\prediction_workplace\\origin.bmp'
MASKPATH = '.\\prediction_workplace\\prediction_threshold.bmp'
MASKEDIMGPATH = '.\\prediction_workplace\\masked.bmp'
PREDICTIONPATH = '.\\prediction_workplace\\prediction.bmp'

def mask_on_image(imgsrcpath, maskpath, maskedImgpath):

    '''put mask on image'''
    
    srcImg = cv2.imread(imgsrcpath, cv2.IMREAD_GRAYSCALE)
    maskImg = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    maskedImg = cv2.bitwise_and(srcImg, srcImg, mask = maskImg)
    cv2.imwrite(maskedImgpath, maskedImg)


def mask_threshold(prediction_path = PREDICTIONPATH):

    mask = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape[0], mask.shape[1]
    etVal, threshold = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(threshold, kernel)
    dilated = cv2.dilate(eroded, kernel)
    Gaussian = cv2.GaussianBlur(dilated, (5, 5), 0)
    contours, hierarchy = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) > 0

    area = []
    #find largest contour
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    black = np.zeros((h, w))
    mask = cv2.drawContours(black, contours, max_idx, 255, cv2.FILLED)

    cv2.namedWindow("mask",0)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

    cv2.imwrite(MASKPATH, mask)
    mask_on_image(IMGSRCPATH, MASKPATH, MASKEDIMGPATH)

    masked = cv2.imread(MASKEDIMGPATH)
    cv2.imshow("mask", masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_single_image()
    mask_threshold()