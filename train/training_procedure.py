from data_loading import *
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cross_validation_training():
    '''use cross-validation to get multiple models'''
    pass

def train_best_model():
    '''decide best model and finally train again'''
    pass

def get_results(path):
    '''give path, predict, get result'''
    pass

