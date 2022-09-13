import os
import sys
import inspect

import torch
from torch._C import dtype
import torch.nn as nn
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from numpy.lib.twodim_base import mask_indices
import tensorflow as tf
#from main import DatasetRSS
from main import DatasetRSS
from model_src.feedbackcontroller import FeedbackController
from model_src.feedbackcontrollerTorch import FeedBackControllerTorch
from model_src.attentionTorch import TopDownAttentionTorch
from model_src.model import PolicyTranslationModel
from model_src.modelTorch import WholeSequenceActor
GLOVE_PATH      = "../GDrive/glove.6B.50d.txt"


class analyzer():
    def __init__(self, path) -> None:
        self.train_data = DatasetRSS(path).ds

    def show_first_entry(self):
        num_elements = 0
        for step, (d_in, d_out) in enumerate(self.train_data):
            #print(len(d_in))
            #print(d_in[2].shape)
            #print(d_in[2][0,2])
            #print(len(d_out))
            #print(d_out[0][0,1])
            #print(d_in[2][0,1:].shape)
            num_elements += 1
        print('num elements')
        print(num_elements)

class tryFeedBackController(tf.keras.Model):
    def __init__(self):
            super(tryFeedBackController, self).__init__(name="tryFeedBackController")
            self.units               = 32
            self.output_dims         = 7
            self.basis_functions     = 11

            self.dout      = tf.keras.layers.Dropout(rate=0.25)

            # Units needs to be divisible by 7
            self.pt_global = tf.keras.layers.Dense(units=42, activation=tf.keras.activations.relu)

            self.pt_dt_1   = tf.keras.layers.Dense(units=self.units * 2, activation=tf.keras.activations.relu)
            self.pt_dt_2   = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.hard_sigmoid)

            self.controller = tf.keras.layers.RNN(
                FeedbackController(
                    robot_state_size = self.units, 
                    rnn_state_size   = (tf.TensorShape([self.output_dims]), tf.TensorShape([self.units])),
                    dimensions       = self.output_dims, 
                    basis_functions  = self.basis_functions,
                    special          = None
                ), 
            return_sequences=True)

    def call_controller(self):
        batch_size = 5
        seq_len = 4
        robot = tf.random.uniform(shape=[batch_size, seq_len, 7], dtype=tf.float32)
        cfeatures = tf.random.uniform(shape=[batch_size, 44], dtype=tf.float32)
        dmp_dt = tf.random.uniform(shape=[batch_size, 1], dtype=tf.float32)
        initial_state = [
            tf.random.uniform(shape=[batch_size, 7], dtype=tf.float32),
            tf.random.uniform(shape=[batch_size, self.units], dtype=tf.float32)
        ]
        generated, phase, weights = self.controller(inputs=robot, constants=(cfeatures, dmp_dt), initial_state=initial_state, training=False)
        print('output RNN')
        print(generated.shape)
        print(phase.shape)
        print(weights.shape)

class tryFeedBackControllerTorch(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.units               = 32
        self.output_dims         = 7
        self.basis_functions     = 11

        self.dout      = tf.keras.layers.Dropout(rate=0.25)

        # Units needs to be divisible by 7

        self.controller = FeedBackControllerTorch(
            robot_state_size=self.units,
            dimensions=self.output_dims,
            basis_functions=self.basis_functions,
            cnfeatures_size=44,
        ).to(device)
        self.device = device

    def call_controller(self):
        batch_size = 1000
        seq_len = 350
        robot = torch.rand(size=(batch_size, seq_len, 7), dtype=torch.float32).to(self.device)
        cfeatures = torch.rand(size=(batch_size, 44), dtype=torch.float32).to(self.device)
        dmp_dt = torch.rand(size=(batch_size, 1), dtype=torch.float32).to(self.device)
        initial_state = [
            torch.rand(size=(batch_size, self.units), dtype=torch.float32).to(self.device),
            torch.rand(size=(batch_size, self.units), dtype=torch.float32).to(self.device)
        ]
        h = time.perf_counter()
        actions_seq, phase_seq, weights_seq = self.controller.forward(seq_inputs=robot, states = initial_state, constants=(cfeatures, dmp_dt))        
        print(f'time in forward: {time.perf_counter() - h}')
        print('output RNN')
        print(actions_seq.shape)
        result = actions_seq.sum()
        h = time.perf_counter()
        result.backward()
        print(f'time for backward: {time.perf_counter() - h}')
        print(phase_seq.shape)
        print(weights_seq.shape)

class try_model():
    def __init__(self) -> None:
        self.modeltf   = PolicyTranslationModel(od_path="", glove_path=GLOVE_PATH)
        self.modeltorch = WholeSequenceActor(od_path="", glove_path=GLOVE_PATH)
        self.d_in_torch = (
            tf.ones((16,5), dtype=tf.float32),
            torch.ones((16,6,5), dtype=torch.float32),
            torch.ones((16,350,7), dtype=torch.float32)
            )

        self.d_in_tf = (
            tf.ones((16,5), dtype=tf.float32),
            tf.ones((16,6,5), dtype=tf.float32),
            tf.ones((16,350,7), dtype=tf.float32)
            )

    def call_model(self):
        #generated, (atn, dmp_dt, phase, weights) = self.modeltf(self.d_in_tf)
        #print('output model')
        #print(generated.shape)
        #print(atn.shape)
        #print(dmp_dt.shape)
        #rint(phase.shape)
        #print(weights.shape)

        generated, (atn, dmp_dt, phase, weights) = self.modeltorch(self.d_in_torch)
        generated, (atn, dmp_dt, phase, weights) = self.modeltorch(self.d_in_torch)
        
        '''print('output model')
        print(generated.shape)
        print(atn.shape)
        print(dmp_dt.shape)
        print(phase.shape)
        print(weights.shape)'''

class tryGRU():
    def __init__(self, device = 'cpu'):
        self.device = device
        self.GRU = nn.GRU(7, 42, device=device)
        self.GRU_cell = nn.GRUCell(7, 42, device = device)

    def call_GRU(self):
        inpt = torch.ones([1000, 350, 7], dtype=torch.float, device=self.device)
        h = time.perf_counter()
        result = self.GRU(inpt)
        print(f'forward time: {time.perf_counter() - h}')
        loss = result[0].sum()
        h = time.perf_counter()
        loss.backward()
        print(f'time in backward: {time.perf_counter() - h}')
        inpt = inpt.transpose(0,1)
        hdn = None
        for event in inpt:
            h = time.perf_counter()
            hdn = self.GRU_cell(event, hdn)
        print(f'forward time: {time.perf_counter() - h}')
        loss = hdn.sum()
        h = time.perf_counter()
        loss.backward()
        print(f'backward time: {time.perf_counter() - h}')


class try_Attention():
    def __init__(self):
        self.attention_model = TopDownAttentionTorch(units=64)

    def callAttention(self):
        language = torch.ones((16, 32))
        features = torch.ones(16,6,5)
        attn = self.attention_model.forward((language, features))
        print(list(self.attention_model.parameters()))
if __name__ == '__main__':
    #ana = analyzer(path = "../GDrive/train.tfrecord")
    #ana.show_first_entry()
    #tfc = tryFeedBackController()
    #tfc.call_controller()
    tfc = tryFeedBackControllerTorch(device='cuda')
    tfc.call_controller()
    #ta = try_Attention()
    #ta.callAttention()
    #tm = try_model()
    #tm.call_model()
    tG = tryGRU(device='cuda')
    tG.call_GRU()