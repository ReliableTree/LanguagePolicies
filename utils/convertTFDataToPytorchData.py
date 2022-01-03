import os
import sys
import inspect

from torch._C import device

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import tensorflow as tf
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader

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
TRAIN_EPOCHS    = 200


from main import DatasetRSS

class TFToTorchConverter():
    def __init__(self, path) -> None:
        self.train_data = DatasetRSS(path, batch_size=1000).ds

    def create_dataset(self, path):
        list_with_torch_files = []
        initialized = False
        #((language, img_ftr, state), (trajectory, onehot, dt, weights, phase, loss_atn))
        for step, data in enumerate(self.train_data):
            num_ele = 0
            for dat in data:
                for ele in dat:
                    ele_torch = torch.tensor(ele.numpy())
                    if not initialized:
                        list_with_torch_files.append(ele_torch)
                    else:
                        list_with_torch_files[num_ele] = torch.cat((list_with_torch_files[num_ele], ele_torch))
                        num_ele += 1
            initialized = True
            print(step * ele_torch.size(0))
            if step > 100:
                break
        print(list_with_torch_files[0].shape)
        with open(path, 'wb') as fp:   #Pickling
            pickle.dump(list_with_torch_files, fp)

    
    def show_first_item(self):
        for step, data in enumerate(self.train_data):
            print(data[0][0][0])
            break
class TorchDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, device = 'cpu', on_device = False, size = 1):
        'Initialization'
        with open(path, 'rb') as fp:   # Unpickling
            self.dataset = pickle.load(fp)
        if on_device:
            ondevice_data = []
            for data in self.dataset:
                ondevice_data.append(data.to(device))
            self.data = ondevice_data
        self.device = device
        self.onDevice = on_device
        


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset[0])

  def __getitem__(self, index):
        'Generates one sample of data'
        if self.onDevice and False:
            return [[dat[index] for dat in self.dataset[:3]], [dat[index] for dat in self.dataset[3:]]]
        else:
            return [[dat[index].to(self.device) for dat in self.dataset[:3]], [dat[index].to(self.device) for dat in self.dataset[3:]]]
        #return ((self.dataset[0][index].to(self.device), self.dataset[1][index].to(self.device), self.dataset[2][index].to(self.device)), (self.dataset[3][index].to(self.device), self.dataset[4][index].to(self.device), self.dataset[5][index].to(self.device), self.dataset[6][index].to(self.device), self.dataset[7][index].to(self.device), self.dataset[8][index].to(self.device)))
#       #return ((language, img_ftr, state), (trajectory, onehot, dt, weights, phase, loss_atn))

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
        TTTC = TFToTorchConverter(path = path_dict['VAL_DATA'])
        TTTC.create_dataset(path = path_dict['VAL_DATA_TORCH'])
        TTTC = TFToTorchConverter(path = path_dict['TRAIN_DATA'])
        TTTC.create_dataset(path = path_dict['TRAIN_DATA_TORCH'])
        #dataset = TorchDataset(path = '../TorchDataset/val_data_torch.txt', device='cuda')
        #dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
        #for step, (d_in, d_out) in enumerate(dataloader):
        #    print(d_in[0].shape)
        #    print(step)
        #print(next(iter(dataloader))[0][2])


