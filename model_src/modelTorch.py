# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import numpy as np
import pathlib

from torch._C import device, dtype
from model_src.attention import TopDownAttention
from model_src.attentionTorch import TopDownAttentionTorch
from model_src.glove import GloveEmbeddings
from model_src.feedbackcontrollerTorch import FeedBackControllerTorch

import torch
import torch.nn as nn
import time

from os import path, makedirs

class PolicyTranslationModelTorch(nn.Module):
    def __init__(self, od_path, glove_path, use_LSTM = False):
        super().__init__()
        self.units               = 32
        self.output_dims         = 7
        self.basis_functions     = 11
        self.ptgloabl_units      = 42

        if od_path != "":                
            od_path    = pathlib.Path(od_path)/"saved_model" 
            self.frcnn = tf.saved_model.load(str(od_path))
            self.frcnn = self.frcnn.signatures['serving_default']
            self.frcnn.trainable = False

        self.embedding = GloveEmbeddings(file_path=glove_path)
        self.lng_gru   = None
        self.dmp_dt_model = None

        self.attention = TopDownAttentionTorch(units=64)

        self.dout      = nn.Dropout(p=0.25)

        # Units needs to be divisible by 7

        self.controller = FeedBackControllerTorch(
            robot_state_size=self.units,
            dimensions=self.output_dims,
            basis_functions=self.basis_functions,
            cnfeatures_size=self.units + self.output_dims + 5,
            use_LSTM = use_LSTM
        )

        self.last_language = None
        self.language = None



    def build_lang_gru(self, input, bias = True):
        self.lng_gru = nn.GRU(input.size(-1), self.units, 1, batch_first = True, device=input.device, bias = bias)


    def build_dmp_dt_model(self, input, use_dropout):
        layers = [
            nn.Linear(input.size(-1), self.ptgloabl_units),
            nn.ReLU()
        ]
        if use_dropout:
            layers.append(nn.Dropout(p=0.25))
        
        layers += [
            nn.Linear(self.ptgloabl_units, self.units * 2),
            nn.ReLU(),
            nn.Linear(self.units * 2, 1),
            nn.Hardsigmoid()
        ]

        self.dmp_dt_model_seq = nn.Sequential(*layers).to(input.device)

        def build_model(inpt):
            dmp_dt = self.dmp_dt_model_seq(inpt)
            return dmp_dt + 0.1

        self.dmp_dt_model = build_model



    def forward(self, inputs, training=False, use_dropout=True, node = None, return_cfeature = False):
        if training:
            use_dropout = True

        language_in   = inputs[0]
        if language_in is not self.last_language:
            self.last_language = language_in

            language_in_tf = tf.convert_to_tensor(language_in.cpu().numpy())
            language  = self.embedding(language_in_tf)

            language = torch.tensor(language.numpy(), device=inputs[1].device)
            if self.lng_gru is None:
                self.build_lang_gru(language)
            _, language  = self.lng_gru(language) 
            self.language = language.squeeze()
        features   = inputs[1]
        # local      = features[:,:,:5]
        robot      = inputs[2]

        # dmp_state  = inputs[3]


        batch_size = robot.size(0)

        # Calculate attention and expand it to match the feature size
        atn = self.attention((self.language, features))
        '''atn_w = atn.unsqueeze(2)
        atn_w = atn_w.repeat([1,1,5])
        # Compress image features and apply attention
        cfeatures = torch.multiply(atn_w, features)
        cfeatures = cfeatures.sum(axis=1)'''

        main_obj = torch.argmax(atn, dim=-1)
        counter = torch.arange(features.size(0))
        cfeatures_max = features[(counter, main_obj)]
        #print(f'atn: {atn[0], atn.shape}')
        #print(f'featues: {features[0], features.shape}')
        #print(f'cfeatures: {cfeatures[0], cfeatures.shape}')
        #print(f'cfeatures_max: {cfeatures_max[0], cfeatures_max.shape}')

        # Add the language to the mix again. Possibly usefull to predict dt
        start_joints  = robot[:,0,:]

        cfeatures = torch.cat((cfeatures_max, self.language, start_joints), axis=1)
        #cfeatures = tf.keras.backend.concatenate((cfeatures, language, start_joints), axis=1)

        # Policy Translation: Create weight + goal for DMP
        if self.dmp_dt_model is None:
            self.build_dmp_dt_model(cfeatures, use_dropout=use_dropout)

        dmp_dt = self.dmp_dt_model(cfeatures)
        if (len(inputs) > 3) and (inputs[3] is not None): #inputs includes Last_GRU_State
            last_gru_state = inputs[3]
        else:
            #last_gru_state = torch.zeros((batch_size, self.units), dtype=torch.float32, device=dmp_dt.device)
            last_gru_state = None

        # Run the low-level controller
        initial_state = [
            start_joints,
            last_gru_state
        ]
        generated, phase, weights, last_gru_state = self.controller.forward(seq_inputs=robot, states=initial_state, constants=(cfeatures, dmp_dt), training=training)

        if return_cfeature:
            return generated, (atn, dmp_dt, phase, weights, cfeatures, last_gru_state)
        else:
            return generated, (atn, dmp_dt, phase, weights)
    
    def getVariables(self, step=None):
        return self.parameters()
    
    def getVariablesFT(self):
        variables = []
        variables += self.pt_w_1.trainable_variables
        variables += self.pt_w_2.trainable_variables
        variables += self.pt_w_3.trainable_variables
        return variables
    
    def saveModelToFile(self, add, data_path):
        import os
        #dir_path = path.dirname(path.realpath(__file__))
        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not path.exists(path_to_file):
            makedirs(path_to_file)
        print(f'saveModelToFile pathL: {path_to_file}')
        torch.save(self.state_dict(), path_to_file + "policy_translation_h")