import os
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from prepocessing import show_img
from unet import UNet

def test_single_image(path = 'sample.bmp'):

    '''predicts a single image'''
    
    net = UNet(n_channels=1, n_classes=1).cuda()
    
    weights = 'unet.pth'

    if os.path.exists(weights):
            net.load_state_dict(torch.load(weights))
            print('weights successfully loaded')
    else:
            print('weights not loaded')
    
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    #image rotation test
    '''
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    '''

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((400,500)), transforms.ToTensor()])

    input = transform(image).cuda()
    input = torch.unsqueeze(input, dim = 0)
    output = net(input)
    save_image(output, 'prediction.bmp')

    img = cv2.imread('prediction.bmp')
    show_img(img)
    print("prediction succesful")

test_single_image()
    


