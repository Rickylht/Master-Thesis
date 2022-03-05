import argparse
from pickletools import optimize
from data_loading import *
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from unet import UNet
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_path = 'unet.pth'
data_path = '.\data'
save_path = '.\data\save_image'
frame = None
frame = AppendtoFrame()


def cross_validation_training():
    '''use cross-validation to get multiple weights'''
    print("Total sample number: ", len(frame))

    transform_objs = transforms.Compose([Random_Shift(), Random_Rotation(), ToPILImage(), ColorJitter(0.5, 0.5, 0.5, 0.3), Rescale((400, 500)), ToTensor(), Normalize()])
    transform_val = transforms.Compose([ToPILImage(), Rescale((400, 500)), ToTensor()])

    #train_val, test = train_test_split(frame, test_size=0.2, random_state=42)
    #train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    transformed_dataset_train = TeethDataset(frame = frame, transform = transform_objs)
    #transformed_dataset_val = TeethDataset(frame = val, transform = transform_val)
    #transformed_dataset_test = TeethDataset(frame = test, transform = transform_val)

    #if out memory, change batch_size smaller
    dataloader_train = DataLoader(transformed_dataset_train, batch_size=1, shuffle=True, num_workers=0)
    #dataloader_val = DataLoader(transformed_dataset_val, batch_size=1, shuffle=True, num_workers=0)
    #dataloader_test = DataLoader(transformed_dataset_test, batch_size=1, shuffle=True, num_workers=0)

    #greyscale, channels = 1
    net = UNet(n_channels=1, n_classes=1).to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successfully load weight')
    else:
        print('not succesfully load weight')
    
    opt = torch.optim.Adam(net.parameters())
    loss = torch.nn.MSELoss()
    
    epoch = 1
    while epoch <= 50:
        for i, sample in enumerate(dataloader_train):
            image, mask = sample['image'].to(device), sample['mask'].to(device)

            out_image = net(image)
            train_loss = loss(out_image, mask)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            #print train loss every 5 times
            if i%5 == 0:
                print(f'{epoch}-{i}-train_loss====>>{train_loss.item()}')

            #save a weights every 20 times
            if i%50 == 0:
                torch.save(net.state_dict(),weight_path)
            
            _image = image[0]
            _mask = mask[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _mask, _out_image], dim = 0)
            save_image(img, f'{save_path}\{i}.bmp')

        epoch += 1


def train_best_model():
    '''decide best model and finally train again'''
    pass



if __name__ == '__main__':
    print('Training start')
    cross_validation_training()
    
    


