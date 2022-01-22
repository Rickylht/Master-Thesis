from data_loading import *
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

frame = None
frame = AppendtoFrame()


def cross_validation_training():
    '''use cross-validation to get multiple models'''
    transform_objs = transforms.Compose([Random_Perspective(), Random_Rotation(), ToPILImage(), Rescale((1024, 1280)), ToTensor()])
    transform_val = transforms.Compose([ToPILImage(), Rescale((1024, 1280)), ToTensor()])

    train_val, test = train_test_split(frame, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    transformed_dataset_train = TeethDataset(frame = train, transform = transform_objs)
    transformed_dataset_val = TeethDataset(frame = val, transform = transform_val)
    transformed_dataset_test = TeethDataset(frame = test, transform = transform_val)

    dataloader_train = DataLoader(transformed_dataset_train, batch_size=5, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(transformed_dataset_val, batch_size=5, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(transformed_dataset_test, batch_size=1, shuffle=True, num_workers=0)

    dataloaders = {'train': dataloader_train, 'val': dataloader_val}



def train_best_model():
    '''decide best model and finally train again'''
    pass

def get_results(path):
    '''give path, predict, get result'''
    pass

