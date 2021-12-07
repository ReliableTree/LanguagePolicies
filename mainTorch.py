# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals

from model_src.modelTorch import PolicyTranslationModelTorch
from utils.networkTorch import NetworkTorch
import hashids
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.convertTFDataToPytorchData import TorchDataset
from prettytable import PrettyTable
import sys



# Learning rate for the adam optimizer
LEARNING_RATE   = 0.0001
# Weight for the attention loss
WEIGHT_ATTN     = 1.0
# Weight for the motion primitive weight loss
WEIGHT_W        = 50.0
# Weight for the trajectroy generation loss
WEIGHT_TRJ      = 1.0
# Weight for the time progression loss
WEIGHT_DT       = 14.0
# Weight for the phase prediction loss
WEIGHT_PHS      = 1.0
# Number of epochs to train
TRAIN_EPOCHS    = 1000


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

def setupModel(device = 'cuda', batch_size = 100, path_dict = None, logname = None, model_path=None):
    model   = PolicyTranslationModelTorch(od_path="", glove_path=path_dict['GLOVE_PATH'], use_LSTM=False).to(device)
    #print(path_dict['TRAIN_DATA_TORCH'])
    train_data = TorchDataset(path = path_dict['TRAIN_DATA_TORCH'], device=device, on_device=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_data = TorchDataset(path = path_dict['VAL_DATA_TORCH'], device=device)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
    network = NetworkTorch(model, data_path=path_dict['DATA_PATH'],logname=logname, lr=LEARNING_RATE, lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS, gamma_sl = 1, device=device)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)

    network.setup_model()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    #init_weights(network)
    count_parameters(network)

    #print(f'number of param,eters in net: {len(list(network.parameters()))} and number of applied: {i}')
    #network.load_state_dict(torch.load(MODEL_PATH), strict=True)
    network.train(epochs=TRAIN_EPOCHS)
    return network
import os
if __name__ == '__main__':
    args = sys.argv[1:]
    if '-path' not in args:
        print('no path given, not executing code')
    else:    
        data_path = args[args.index('-path') + 1]
        path_dict = {
        'TRAIN_DATA_TORCH' : os.path.join(data_path, 'TorchDataset/train_data_torch.txt'),
        'VAL_DATA_TORCH' : os.path.join(data_path, 'TorchDataset/val_data_torch.txt'),
        'MODEL_PATH' : os.path.join(data_path, 'TorchDataset/test_model.pth'),
        'TRAIN_DATA' : os.path.join(data_path, 'GDrive/train.tfrecord'),
        'VAL_DATA' : os.path.join(data_path, 'GDrive/validate.tfrecord'),
        'GLOVE_PATH' : os.path.join(data_path, 'GDrive/glove.6B.50d.txt'),
        'DATA_PATH' : data_path
        }

        device = 'cuda'
        if '-device' in args:
            device = args[args.index('-device') + 1]

        model_path = None
        if '-model' in args:
            model_path = args[args.index('-model') + 1]

        hid             = hashids.Hashids()
        logname         = hid.encode(int(time.time() * 1000000))
        network = setupModel(device=device, batch_size = 32, path_dict = path_dict, logname=logname, model_path=model_path)
        print(f'end saving: {path_dict["MODEL_PATH"]}')
        torch.save(network.state_dict(), path_dict['MODEL_PATH'])

