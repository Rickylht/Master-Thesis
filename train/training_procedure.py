import argparse
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

writer = SummaryWriter('.\\tb_evaluation\\log')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_path = 'unet.pth'
data_path = '.\\data'
save_path = '.\\data\\save_image'
frame = None
frame = AppendtoFrame()


def check_integrity():
    pass

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = torch.nn.functional.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = torch.nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def cross_validation_training():
    '''use cross-validation to get multiple weights'''
    print('------Training start------\n')

    print("Device: ", device)

    print("Total sample number: ", len(frame))

    time_start = time.time()
    #transformation
    transform_objs = transforms.Compose([Random_Shift(), Random_Rotation(), ToPILImage(), ColorJitter(0.5, 0.5, 0.5, 0.3), Rescale((400, 500)), ToTensor(), Normalize()])
    transform_val = transforms.Compose([ToPILImage(), Rescale((400, 500)), ToTensor(), Normalize()])

    train, val = train_test_split(frame, test_size=0.2, random_state=42)

    transformed_dataset_train = TeethDataset(frame = train, transform = transform_objs)
    transformed_dataset_val = TeethDataset(frame = val, transform = transform_val)

    #if out memory, change batch_size smaller
    dataloader_train = DataLoader(transformed_dataset_train, batch_size=2, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(transformed_dataset_val, batch_size=2, shuffle=True, num_workers=0)

    #greyscale -> channels = 1, 2 classes
    net = UNet(n_channels=1, n_classes=1).to(device)

    '''
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successfully load weight')
    else:
        print('not succesfully load weight')
    '''

    opt = torch.optim.Adam(net.parameters())
    loss = torch.nn.BCEWithLogitsLoss()

    count_train = 0
    count_eval = 0

    train_loss_list = []
    eval_loss_list = []

    train_iteration_list = []
    eval_iteration_list = []


    num_epochs = 15
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

            writer.add_scalar("Loss/Train", train_loss, count_train)
            
            #print loss every 5 times
            if i%5 == 0:
                print(f'{epoch}-{i}-train_loss====>>{train_loss.item()}')
                
                #decide evaluation
                net.eval()

                for i, sample in enumerate(dataloader_val):
                    eval_image, eval_mask = sample['image'].to(device), sample['mask'].to(device)
                    with torch.no_grad():
                        out_image = net(eval_image)
                        eval_loss = loss(out_image, eval_mask)
                        eval_loss_list.append(eval_loss.data)
                    count_eval += 1
                    eval_iteration_list.append(count_eval)
                    writer.add_scalar("Loss/Evaluation", eval_loss, count_eval)

                net.train()

            _image = image[0]
            _mask = mask[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _mask, _out_image], dim = 0)
            save_image(img, f'{save_path}\{i}.bmp')
    
    torch.save(net.state_dict(),weight_path)

    time_end = time.time()

    plt.plot(train_iteration_list,train_loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Unet: train Loss vs Number of iteration")
    plt.savefig('.\\tb_evaluation\\train_loss.jpg')
    plt.show()

    plt.plot(eval_iteration_list,eval_loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Unet: eval Loss vs Number of iteration")
    plt.savefig('.\\tb_evaluation\\eval_loss.jpg')
    plt.show()

    print('Time cost: ',time_end - time_start,'s')
    print("\n------Training end------")


if __name__ == '__main__':
    cross_validation_training()
    
    


