import os
import shutil
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from unet import UNet
import numpy as np

def test_single_image(path = '.\\data\\imgs\\830_lingual.bmp'):

    '''predict a single image'''

    shutil.copy(path, '.\\prediction_workplace\\origin.bmp')
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    net = UNet(n_channels=1, n_classes=1).cuda()
    
    weights = 'unet.pth'

    if os.path.exists(weights):
            net.load_state_dict(torch.load(weights))
    else:
            print('weights not loaded')
            return
    
    h, w = image.shape[0], image.shape[1]

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((400,500)), transforms.ToTensor()])

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

if __name__ == '__main__':
    test_single_image()