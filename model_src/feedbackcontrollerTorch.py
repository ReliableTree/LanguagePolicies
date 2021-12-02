# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from model_src.basismodelTorch import BasisModelTorch
from model_src.basismodel import BasisModel

import torch
import torch.nn as nn
import time


    

class FeedBackControllerCellTorch(nn.Module):
    def __init__(self, robot_state_size, dimensions, basis_functions, cnfeatures_size = 5, bias = True, use_LSTM = False):
        super().__init__()
        self.robot_state_size = robot_state_size
        self.dims             = dimensions
        self.n_bfuncs         = basis_functions
        self.x_shape = robot_state_size + cnfeatures_size

        if not use_LSTM:
            self.robot_gru = nn.GRUCell(input_size=self.dims, hidden_size=self.robot_state_size, bias= bias)

        else:
            self.robot_gru = nn.LSTMCell(input_size=self.dims, hidden_size=self.robot_state_size, bias= bias)

        self.kin_model = nn.Sequential(
            nn.Linear(self.x_shape, self.dims * self.n_bfuncs),
            nn.ReLU(),
            nn.Linear(self.dims * self.n_bfuncs, self.dims * self.n_bfuncs),
            nn.ReLU(),
            nn.Linear(self.dims * self.n_bfuncs, self.dims * self.n_bfuncs),
        )

        self.phase_model = nn.Sequential(
            nn.Linear(self.x_shape, int(self.robot_state_size / 2.0)),
            nn.ReLU(),
            nn.Linear(int(self.robot_state_size / 2.0), 1),
            nn.Hardsigmoid()
        )

        self.basismodel = BasisModelTorch(nfunctions=self.n_bfuncs, scale=0.012)



    def forward(self, inputs, states, constants=None, training=False, mask=None, **kwargs):
        # Get data ready
        in_robot       = inputs
        st_robot_last  = states[0]
        st_gru_last    = states[1]
        cn_features    = constants[0]
        cn_delta_t     = constants[1]
        
        # Robot GRU:
        if training:
            in_robot = st_robot_last

        #h = time.perf_counter()
        if st_robot_last is not None:
            gru_output = self.robot_gru(in_robot, st_gru_last)
        else:
            gru_output = self.robot_gru(in_robot)
        #print(f'time for one cell call: {time.perf_counter() - h}')
        # Internal state:
        if type(gru_output) is list:
            x = torch.cat((cn_features, gru_output[0]), axis=1)
        else:
            x = torch.cat((cn_features, gru_output), axis=1)

        # Use x to calcate the weights:

        weights = self.kin_model(x).reshape([-1, self.dims, self.n_bfuncs])

        # Phase estimation, based on x:
        dt    = 1.0 / (500.0 * cn_delta_t) # Calculates the actual dt
        phase = self.phase_model(x)
        phase = phase + dt

        # Apply basis model:
        action = self.basismodel.apply_basis_model((weights, phase))
        action = torch.squeeze(action)

        return (action, phase, weights), gru_output


class FeedBackControllerTorch(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.Cell = FeedBackControllerCellTorch(**kwargs)
        
    def forward(self, seq_inputs, states, constants=None, training=False):
        actions_seq = None
        phase_seq = None
        weights_seq = None
        initilaized = False
        for event in range(int(seq_inputs.size(1))):
            (action, phase, weights), gru_output = self.Cell.forward(inputs=seq_inputs[:,event,:], states=states, constants=constants, training=training)
            states = (action, gru_output)
            if not initilaized:
                actions_seq = torch.zeros_like(seq_inputs)
                phase_seq = torch.zeros([seq_inputs.size(0), seq_inputs.size(1), 1], device = seq_inputs.device, dtype=seq_inputs.dtype)
                weights_seq = torch.zeros([seq_inputs.size(0), seq_inputs.size(1), weights.size(1), weights.size(2)], device = seq_inputs.device, dtype=seq_inputs.dtype)
                initilaized = True
                
            actions_seq[:,event,:] = action
            phase_seq[:,event,:] = phase
            weights_seq[:,event,:] = weights
        return actions_seq, phase_seq, weights_seq, gru_output