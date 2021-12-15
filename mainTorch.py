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
import pickle



# Learning rate for the adam optimizer
LEARNING_RATE   = 0.0001
# Weight for the attention loss
WEIGHT_ATTN     = 1.0
# Weight for the motion primitive weight loss
WEIGHT_W        = 50.0
# Weight for the trajectroy generation loss
WEIGHT_TRJ      = 50#5.0

WEIGHT_GEN_TRJ  = 50

# Weight for the time progression loss
WEIGHT_DT       = 14.0
# Weight for the phase prediction loss
WEIGHT_PHS      = 50 #1.0

WEIGHT_FOD      = 0

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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

def setupModel(device , epochs ,  batch_size, path_dict , logname , model_path, tboard, model_setup):
    model   = PolicyTranslationModelTorch(od_path="", glove_path=path_dict['GLOVE_PATH'], model_setup=model_setup).to(device)
    #print(path_dict['TRAIN_DATA_TORCH'])
    train_data = TorchDataset(path = path_dict['TRAIN_DATA_TORCH'], device=device, on_device=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_data = TorchDataset(path = path_dict['VAL_DATA_TORCH'], device=device)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
    network = NetworkTorch(model, data_path=path_dict['DATA_PATH'],logname=logname, lr=LEARNING_RATE, lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_gen_trj = WEIGHT_GEN_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS, lw_fod=WEIGHT_FOD, gamma_sl = 1, device=device, tboard=tboard)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)

    network.setup_model(model_params=model_setup)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    #init_weights(network)
    count_parameters(network)

    #print(f'number of param,eters in net: {len(list(network.parameters()))} and number of applied: {i}')
    #network.load_state_dict(torch.load(MODEL_PATH), strict=True)
    network.train(epochs=epochs, model_params=model_setup)
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
        'TRAIN_DATA' : os.path.join(data_path, 'GDrive/train.tfrecord'),
        'VAL_DATA' : os.path.join(data_path, 'GDrive/validate.tfrecord'),
        'GLOVE_PATH' : os.path.join(data_path, 'GDrive/glove.6B.50d.txt'),
        'DATA_PATH' : data_path
        }

        device = 'cuda'
        if '-device' in args:
            device = args[args.index('-device') + 1]

        model_setup = {
            'obj_embedding': {'use_obj_embedding':True, 'train_embedding':True, 'EIS':30, 'EOS':10},
            'attn_trans' : {'use_attn_trans':True},
            'lang_trans' :  {
                'use_lang_trans' : True,
                'd_output' : 32,
                'd_model'  : 42,
                'nhead'    : 1,
                'nlayers'  : 1
            },
            'contr_trans': {
                'use_contr_trans':True,
                'd_output'   : 8,
                'd_model'    : 210,
                'nhead'      : 6,
                'nlayers'    : 4,
                'recursive'    : False,
                'use_gen2'     : False,
                'use_mask'     : False,
                'use_counter_embedding': False,
                'count_emb_dim' : 20,
                'plan_nn'       : {
                    'use_plan_nn'   : True,
                    'plan'     :{
                        'use_layernorm':False,
                        'plan_type' : 'upconv',
                        'num_upconvs':5,
                        'stride':3,
                        'd_output':8,
                        'nhead':8,
                        'd_hid':80,
                        'nlayers':3
                    },
                }

            },
            'LSTM':{
                'use_LSTM' : False
            },
            'quick_val':False,
            'val_every' : 1
        }
        model_path = None
        if '-model' in args:
            model_path = args[args.index('-model') + 1] + 'policy_translation_h'
            if '-model_setup' in args:
                setup_path = args[args.index('-model') + 1] + 'model_setup.pkl'
                with open(setup_path, 'rb') as f:
                    model_setup = pickle.load(f)
                print('load model')

        epochs = 200
        if '-epochs' in args:
            epochs = int(args[args.index('-epochs') + 1])

        batch_size = 16
        if '-batch_size' in args:
            batch_size = int(args[args.index('-batch_size') + 1])

        tboard = True
        if '-tboard' in args:
            tboard = (args[args.index('-tboard') + 1]) == 'True'
            print(f'tboard: {tboard}')

        hid             = hashids.Hashids()
        logname         = hid.encode(int(time.time() * 1000000))
        print(f'logname: {logname}')
        network = setupModel(device=device, epochs = epochs, batch_size = batch_size, path_dict = path_dict, logname=logname, model_path=model_path, tboard=tboard, model_setup=model_setup)
        print(f'end saving: {path_dict["MODEL_PATH"]}')
        torch.save(network.state_dict(), path_dict['MODEL_PATH'])

