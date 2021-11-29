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
WEIGHT_TRJ      = 5.0
# Weight for the time progression loss
WEIGHT_DT       = 14.0
# Weight for the phase prediction loss
WEIGHT_PHS      = 1.0
# Number of epochs to train
TRAIN_EPOCHS    = 1000

import pickle
def load_tf_statedict(model):
    with open('paras_tf.pkl', 'rb') as f:
        paras_tf = pickle.load(f)
    torch_state_dict = model.state_dict()
    #Attention
    torch_state_dict['attention.w1.0.weight'] = torch.tensor(paras_tf['attention/time_distributed/kernel:0']).T
    torch_state_dict['attention.w1.0.bias'] = torch.tensor(paras_tf['attention/time_distributed/bias:0']).T
    torch_state_dict['attention.w2.0.weight'] = torch.tensor(paras_tf['attention/time_distributed_1/kernel:0']).T
    torch_state_dict['attention.w1.0.bias'] = torch.tensor(paras_tf['attention/time_distributed_1/bias:0']).T
    torch_state_dict['attention.wt.weight'] = torch.tensor(paras_tf['attention/time_distributed_2/kernel:0']).T

    #LanguageGRU
    torch_state_dict['lng_gru.weight_ih_l0'] = torch.tensor(paras_tf['gru/kernel:0']).T
    torch_state_dict['lng_gru.weight_hh_l0'] = torch.tensor(paras_tf['gru/recurrent_kernel:0']).T
    torch_state_dict['lng_gru.bias_ih_l0'] = torch.tensor(paras_tf['gru/bias:0'][1]).T
    torch_state_dict['lng_gru.bias_hh_l0'] = torch.tensor(paras_tf['gru/bias:0'][0]).T

    #Controller GRU
    torch_state_dict['controller.Cell.robot_gru.weight_ih'] = torch.tensor(paras_tf['rnn/gru_cell/kernel:0']).T
    torch_state_dict['controller.Cell.robot_gru.weight_hh'] = torch.tensor(paras_tf['rnn/gru_cell/recurrent_kernel:0']).T
    torch_state_dict['controller.Cell.robot_gru.bias_ih'] = torch.tensor(paras_tf['rnn/gru_cell/bias:0'][1]).T
    torch_state_dict['controller.Cell.robot_gru.bias_hh'] = torch.tensor(paras_tf['rnn/gru_cell/bias:0'][0]).T

    #ControllerKinModel
    torch_state_dict['controller.Cell.kin_model.0.weight'] = torch.tensor(paras_tf['rnn/dense_3/kernel:0']).T
    torch_state_dict['controller.Cell.kin_model.0.bias'] = torch.tensor(paras_tf['rnn/dense_3/bias:0']).T
    torch_state_dict['controller.Cell.kin_model.2.weight'] = torch.tensor(paras_tf['rnn/dense_4/kernel:0']).T
    torch_state_dict['controller.Cell.kin_model.2.bias'] = torch.tensor(paras_tf['rnn/dense_4/bias:0']).T
    torch_state_dict['controller.Cell.kin_model.4.weight'] = torch.tensor(paras_tf['rnn/dense_5/kernel:0']).T
    torch_state_dict['controller.Cell.kin_model.4.bias'] = torch.tensor(paras_tf['rnn/dense_5/bias:0']).T

    #ControllerPhaseModel
    torch_state_dict['controller.Cell.phase_model.0.weight'] = torch.tensor(paras_tf['rnn/dense_6/kernel:0']).T
    torch_state_dict['controller.Cell.phase_model.0.bias'] = torch.tensor(paras_tf['rnn/dense_6/bias:0']).T
    torch_state_dict['controller.Cell.phase_model.2.weight'] = torch.tensor(paras_tf['rnn/dense_7/kernel:0']).T
    torch_state_dict['controller.Cell.phase_model.2.bias'] = torch.tensor(paras_tf['rnn/dense_7/bias:0']).T

    #dmpdt model
    torch_state_dict['dmp_dt_model_seq.0.weight'] = torch.tensor(paras_tf['dense/kernel:0']).T
    torch_state_dict['dmp_dt_model_seq.0.bias'] = torch.tensor(paras_tf['dense/bias:0']).T
    torch_state_dict['dmp_dt_model_seq.3.weight'] = torch.tensor(paras_tf['dense_1/kernel:0']).T
    torch_state_dict['dmp_dt_model_seq.3.bias'] = torch.tensor(paras_tf['dense_1/bias:0']).T
    torch_state_dict['dmp_dt_model_seq.5.weight'] = torch.tensor(paras_tf['dense_2/kernel:0']).T
    torch_state_dict['dmp_dt_model_seq.5.bias'] = torch.tensor(paras_tf['dense_2/bias:0'])


    model.load_state_dict(torch_state_dict, strict=True)
    print('loaded state dict')
    return model


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

def setupModel(device = 'cuda', batch_size = 16, path_dict = None, logname = None, model_path=None):
    print("  --> Running with default settings")
    model   = PolicyTranslationModelTorch(od_path="", glove_path=path_dict['GLOVE_PATH'], use_LSTM=False).to(device)
    '''train_data = TorchDataset(path = path_dict['TRAIN_DATA_TORCH'], device=device, on_device=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)


    eval_data = TorchDataset(path = path_dict['VAL_DATA_TORCH'], device=device)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    network = NetworkTorch(model, data_path=path_dict['DATA_PATH'],logname=logname, lr=LEARNING_RATE, lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS, gamma_sl = 1, device=device)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)
    network.setup_model()'''
    #if model_path is not None:
    #    model.load_state_dict(torch.load(model_path))
    bs = 2

    model((
    torch.ones((bs, 15), dtype=torch.int64),
    torch.ones((bs, 6, 5), dtype=torch.float32),
    torch.ones((bs, 500, 7), dtype=torch.float32)
    ))
    model = load_tf_statedict(model)
    print('before')
    generated, (atn, dmp_dt, phase, weights) = model((
        torch.ones((bs, 15), dtype=torch.int64),
        torch.ones((bs, 6, 5), dtype=torch.float32),
        torch.ones((bs, 500, 7), dtype=torch.float32)
    ))
    #print(generated[0,0])
    #print(generated[0,0].shape)

    #print(f'number of param,eters in net: {len(list(network.parameters()))} and number of applied: {i}')
    #network.load_state_dict(torch.load(MODEL_PATH), strict=True)
    #network.train(epochs=TRAIN_EPOCHS)
    return model
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

        model_path = None
        if '-model' in args:
            model_path = args[args.index('-model') + 1]

        hid             = hashids.Hashids()
        logname         = hid.encode(int(time.time() * 1000000))
        #network = setupModel(device='cuda', batch_size = 1000, path_dict = path_dict, logname=logname, model_path=model_path)
        model = setupModel(device='cpu', batch_size = 16, path_dict = path_dict, logname=logname, model_path=model_path)
        #print(f'end saving: {path_dict["MODEL_PATH"]}')
        #torch.save(network.state_dict(), path_dict['MODEL_PATH'])

