import os
import sys
import inspect

import torch
import torch.nn as nn

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from numpy.lib.twodim_base import mask_indices
import tensorflow as tf
from main import DatasetRSS
from model_src.feedbackcontroller import FeedbackController
from model_src.feedbackcontrollerTorch import FeedBackControllerTorch

from model_src.attentionTorch import TopDownAttentionTorch

class analyzer():
    def __init__(self, path) -> None:
        self.train_data = DatasetRSS(path).ds

    def show_first_entry(self):
        for step, (d_in, d_out) in enumerate(self.train_data):
            print(len(d_in))
            #print(d_in[2].shape)
            #print(d_in[2][0,2])
            #print(len(d_out))
            #print(d_out[0][0,1])
            print(d_in[2][0,1:].shape)
            break

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
    def __init__(self):
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
            )

    def call_controller(self):
        batch_size = 16
        seq_len = 350
        robot = torch.rand(size=(batch_size, seq_len, 7), dtype=torch.float32)
        cfeatures = torch.rand(size=(batch_size, 44), dtype=torch.float32)
        dmp_dt = torch.rand(size=(batch_size, 1), dtype=torch.float32)
        initial_state = [
            torch.rand(size=(batch_size, self.units), dtype=torch.float32),
            torch.rand(size=(batch_size, self.units), dtype=torch.float32)
        ]
        actions_seq, phase_seq, weights_seq = self.controller.forward(seq_inputs=robot, states = initial_state, constants=(cfeatures, dmp_dt))        
        
        print('output RNN')
        print(actions_seq.shape)
        print(phase_seq.shape)
        print(weights_seq.shape)

class try_Attention():
    def __init__(self):
        self.attention_model = TopDownAttentionTorch(units=64)

    def callAttention(self):
        language = torch.ones((16, 32))
        features = torch.ones(16,6,5)
        self.attention_model.forward((language, features))

if __name__ == '__main__':
    #ana = analyzer(path = "../GDrive/train.tfrecord")
    #ana.show_first_entry()
    #tfc = tryFeedBackController()
    #tfc.call_controller()
    #tfc = tryFeedBackControllerTorch()
    #tfc.call_controller()
    ta = try_Attention()
    ta.callAttention()
