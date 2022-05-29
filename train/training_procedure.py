from pickle import FRAME
from pickletools import optimize
from data_loading import *
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from unet import UNet
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter  
import time
from torchvision import transforms

# Set flag = 0 for contour segmentation, flag = 1 for caries estimation
# Don`t forget to change it in 'data_loading.py'
flag = 1

if flag == 0:
    weight_path = '.\\weights\\seg_unet{}.pth'
    save_path = '.\\data\\seg_save_image'
else:
    weight_path = '.\\weights\\caries_unet{}.pth'
    save_path = '.\\data\\caries_save_image'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
frame = None
frame = AppendtoFrame()
frame = frame.sample(frac=1).reset_index(drop=True)

def cross_validation_training():
    '''Cross-validation to get multiple weights'''

    if flag == 0:
        print('\n------Contour segmentation------')
    else:
        print('\n------Caries estimation------')
    
    print('\n------Cross Validation Start------')

    print("\nDevice: ", device)

    print("\nTotal sample number: ", len(frame))

    print("\nFirst ten rows of dataframe: ")
    print(frame.head(10))

    print("\nLoss visuailization: \n")
    
    #transformation
    transform_objs = transforms.Compose([Random_Shift(), Random_Rotation(), ToPILImage(), ColorJitter(0.5, 0.5, 0.5, 0.3), Rescale((400, 500)), ToTensor(), Normalize()])
    transform_val = transforms.Compose([ToPILImage(), Rescale((400, 500)), ToTensor(), Normalize()])

    i = 1

    plt.figure(figsize=(16,8))
    # CV
    for n in range(1,6):

        time_start = time.time()

        if n == 1 :
            train = frame.loc[0:63]
            val = frame.loc[64:79]
        if n == 2 :
            train = frame.loc[16:79]
            val = frame.loc[0:15]
        if n == 3 :
            train = pd.concat([frame.loc[0:15], frame.loc[32:79]])
            val = frame.loc[16:31]
        if n == 4 :
            train = pd.concat([frame.loc[0:31], frame.loc[48:79]])
            val = frame.loc[32:47]
        if n == 5 :
            train = pd.concat([frame.loc[0:47], frame.loc[64:79]])
            val = frame.loc[48:63]
    
        transformed_dataset_train = TeethDataset(frame = train, transform = transform_objs)
        transformed_dataset_val = TeethDataset(frame = val, transform = transform_val)

        #if out of memory, change batch_size smaller
        dataloader_train = DataLoader(transformed_dataset_train, batch_size=2, shuffle=True, num_workers=0)
        dataloader_val = DataLoader(transformed_dataset_val, batch_size=2, shuffle=True, num_workers=0)

        #greyscale -> channels = 1, 2 classes
        net = UNet(n_channels=1, n_classes=1).to(device)

        opt = torch.optim.Adam(net.parameters())
        #loss = torch.nn.MSELoss()
        loss = torch.nn.BCEWithLogitsLoss()

        count_train = 0
        count_val = 0

        train_iteration_list = []
        val_iteration_list = []

        train_loss_list = []
        val_loss_list = []

        num_epochs = 5 #only for test
        for epoch in range(num_epochs):
            net.train()
            for i, sample in enumerate(dataloader_train):
                image, mask = sample['image'].to(device), sample['mask'].to(device)

                out_image = net(image)
                train_loss = loss(out_image, mask)

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                count_train += 1
                train_loss_list.append(train_loss.data)
                train_iteration_list.append(count_train)
            
                #print loss every 5 times
                if i%5 == 0:
                    print(f'{epoch}-{i}-train_loss====>>{train_loss.item()}')

                    net.eval()
                    for i, sample in enumerate(dataloader_val):
                        val_image, val_mask = sample['image'].to(device), sample['mask'].to(device)
                        with torch.no_grad():
                            out_image = net(val_image)
                            val_loss = loss(out_image, val_mask)
                            val_loss_list.append(val_loss.data)
                        count_val += 1
                        val_iteration_list.append(count_val)
                    net.train()
    
        torch.save(net.state_dict(), weight_path.format(n))

        time_end = time.time()

        if flag == 0:
            traintitle = "Seg: train Loss vs iteration"
            valtitle = "Seg: val Loss vs iteration"

        else :
            traintitle = "Caries: train Loss vs iteration"
            valtitle = "Caries: val Loss vs iteration"
            
        plt.subplot(1,2,1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.plot(train_iteration_list,train_loss_list)
        plt.title(traintitle)

        plt.subplot(1,2,2)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.plot(val_iteration_list,val_loss_list)
        plt.title(valtitle)

        print('Time cost: ',time_end - time_start,'s')
        print("\n", n," fold ends")
    
    plt.savefig(".\\tensorboard_eval\\seg_log\\seg.jpg")
    print("\n------Training end------")


def train_best():
    '''train best weights with all data'''

    print('\n------Train Best Start------')
    print("\nDevice: ", device)
    print("\nTotal sample number: ", len(frame))
    print("\nFirst ten rows of dataframe: ")
    print(frame.head(10))
    print("\nLoss visuailization: \n")

    time_start = time.time()

    #transformation
    transform_objs = transforms.Compose([Random_Shift(), Random_Rotation(), ToPILImage(), ColorJitter(0.5, 0.5, 0.5, 0.3), Rescale((400, 500)), ToTensor(), Normalize()])
    transformed_dataset_train = TeethDataset(frame = frame, transform = transform_objs)

    #if out memory, change batch_size smaller
    dataloader_train = DataLoader(transformed_dataset_train, batch_size=2, shuffle=True, num_workers=0)

    #greyscale -> channels = 1, 2 classes
    net = UNet(n_channels=1, n_classes=1).to(device)

    opt = torch.optim.Adam(net.parameters())
    #loss = torch.nn.MSELoss()
    loss = torch.nn.BCEWithLogitsLoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        net.train()
        for i, sample in enumerate(dataloader_train):
            image, mask = sample['image'].to(device), sample['mask'].to(device)

            out_image = net(image)
            train_loss = loss(out_image, mask)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%5 == 0:
                    print(f'{epoch}-{i}-train_loss====>>{train_loss.item()}')
            
            #visualization 
            _image = image[0]
            _mask = mask[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _mask, _out_image], dim = 0)
            save_image(img, f'{save_path}\{i}.bmp')
    
    if flag == 0:
        weights = "best_seg.pth"
    else:
        weights = "best_caries.pth"

    torch.save(net.state_dict(), weights)
    time_end = time.time()

    print('Time cost: ',time_end - time_start,'s')
    print("\n------Training end------")
    
if __name__ == '__main__':
    #cross_validation_training()
    #train_best()
    pass
