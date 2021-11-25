# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals
from pickle import load

from utils.tf_util import limitGPUMemory, trainOnCPU
from model_src.model import PolicyTranslationModel
from utils.network import Network
from model_src.modelTorch import PolicyTranslationModelTorch
from utils.networkTorch import NetworkTorch
import tensorflow as tf
import hashids
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.convertTFDataToPytorchData import TorchDataset
from prettytable import PrettyTable


TRAIN_DATA_TORCH = '../TorchDataset/train_data_torch.txt'

VAL_DATA_TORCH = '../TorchDataset/val_data_torch.txt'

MODEL_PATH = '../TorchDataset/test_init_model.pth'


# Location of the training data
TRAIN_DATA      = "../GDrive/train.tfrecord"
# Location of the validation data
VALIDATION_DATA = "../GDrive/validate.tfrecord"
# Location of the GloVe word embeddings
GLOVE_PATH      = "../GDrive/glove.6B.50d.txt"
# Learning rate for the adam optimizer
LEARNING_RATE   = 0.0001
# Weight for the attention loss
WEIGHT_ATTN     = 1.0
# Weight for the motion primitive weight loss
WEIGHT_W        = 50.0
# Weight for the trajectroy generation loss
WEIGHT_TRJ      = 5.0
# Weight for the time progression loss
WEIGHT_DT       = 14.0
# Weight for the phase prediction loss
WEIGHT_PHS      = 1.0
# Number of epochs to train
TRAIN_EPOCHS    = 100
hid             = hashids.Hashids()
LOGNAME         = hid.encode(int(time.time() * 1000000))

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def init_weights(network):
    nw_paras = network.named_parameters()
    for para_name, para in nw_paras:
        if 'bias' in para_name:
            para.data.fill_(0.01)

        elif 'weight' in para_name:
            torch.nn.init.orthogonal_(para)

def setupModel(device = 'cuda', batch_size = 16):
    print("  --> Running with default settings")
    model   = PolicyTranslationModelTorch(od_path="", glove_path=GLOVE_PATH, use_LSTM=False).to(device)
    train_data = TorchDataset(path = TRAIN_DATA_TORCH, device=device, on_device=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)


    eval_data = TorchDataset(path = VAL_DATA_TORCH, device=device)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    network = NetworkTorch(model, logname=LOGNAME, lr=LEARNING_RATE, lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS, gamma_sl = 1, device=device)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)
    network.setup_model()
    #init_weights(network)
    count_parameters(network)

    #print(f'number of param,eters in net: {len(list(network.parameters()))} and number of applied: {i}')
    #network.load_state_dict(torch.load(MODEL_PATH), strict=True)
    network.train(epochs=TRAIN_EPOCHS)
    return network

if __name__ == '__main__':
    import pickle
    hid             = hashids.Hashids()
    LOGNAME         = hid.encode(int(time.time() * 1000000))
    network = setupModel(device='cpu')
    #torch.save(network.state_dict(), MODEL_PATH)

