# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals

from torch._C import device

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
LEARNING_RATE   = 1e-4
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

#torch.set_default_dtype(torch.float64)
def init_weights(network):
    nw_statedict = network.state_dict()
    for para in nw_statedict:
        print(para)
        if 'bias' in para:
            nw_statedict[para].data.fill_(1e-4)
        elif 'weight' in para:
            torch.nn.init.orthogonal_(nw_statedict[para])
        else:
            print(para)

def setupModel(device = 'cuda', batch_size = 1000):
    print("  --> Running with default settings")
    model   = PolicyTranslationModelTorch(od_path="", glove_path=GLOVE_PATH).to(device)
    train_data = TorchDataset(path = TRAIN_DATA_TORCH, device=device, on_device=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


    eval_data = TorchDataset(path = VAL_DATA_TORCH, device=device)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
    network = NetworkTorch(model, logname=LOGNAME, lr=LEARNING_RATE, lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)
    network.setup_model()
    i = 0
    init_weights(network)

    print(f'number of param,eters in net: {len(list(network.parameters()))} and number of applied: {i}')
    #network.load_state_dict(torch.load(MODEL_PATH), strict=True)
    network.train(epochs=TRAIN_EPOCHS)
    return network

if __name__ == '__main__':
    trainOnCPU()
    hid             = hashids.Hashids()
    LOGNAME         = hid.encode(int(time.time() * 1000000))
    network = setupModel(device='cuda')
    torch.save(network.state_dict(), MODEL_PATH)
    # model.summary()

