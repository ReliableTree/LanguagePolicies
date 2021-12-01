# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import numpy as np
import pathlib

from torch._C import device, dtype
from model_src.attention import TopDownAttention
from model_src.attentionTorch import TopDownAttentionTorch
from model_src.glove import GloveEmbeddings
from model_src.feedbackcontrollerTorch import FeedBackControllerTorch

from JupyterTryOut.LangGruSetup import set_up_GRU_paras_torch
from JupyterTryOut.ATTSetup import load_Att_Torch
from JupyterTryOut.dmpSetup import load_torch_dmp

import torch
import torch.nn as nn
import time

from os import path, makedirs

class myHardSigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        result = input
        result_size = result.size()
        result = result.view(-1)
        result[result < -2.5] = 0
        result[result > 2.5] = 1
        result[(result >= -2.5) & (result <= 2.5)] = 0.2*result+0.5
        result = result.view(*result_size)
        return result

class dmp_dt_torch(nn.Module):
    def __init__(self, ptgloabl_units, units) -> None:
        super().__init__()
        self.ptgloabl_units = ptgloabl_units
        self.units = units


    def build_dmp_dt_model(self, input):
            pre_layers = [
                nn.Linear(input.size(-1), self.ptgloabl_units),
                nn.ReLU()
            ]
            
            post_layers = [
                nn.Linear(self.ptgloabl_units, self.units * 2),
                nn.ReLU(),
                nn.Linear(self.units * 2, 1),
                myHardSigmoid()
            ]

            self.dmp_dt_model_pre = nn.ModuleList(pre_layers).to(input.device)
            self.dmp_dt_model_post = nn.ModuleList(post_layers).to(input.device)

            def build_model(inpt, use_dropout):
                result = inpt
                for l in self.dmp_dt_model_pre:
                    result = l(result)

                if use_dropout:
                    print('used dropout')
                    result = nn.Dropout(p=0.25)(result)

                for l in self.dmp_dt_model_post:
                    result = l(result)
                return result + 0.1

            self.dmp_model = build_model

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



    def build_lang_gru(self, input, bias = True):
        self.lng_gru = nn.GRU(input.size(-1), self.units, 1, batch_first = True, device=input.device, bias = bias)



    '''def build_dmp_dt_model(self, input):
        pre_layers = [
            nn.Linear(input.size(-1), self.ptgloabl_units),
            nn.ReLU()
        ]
        
        post_layers = [
            nn.Linear(self.ptgloabl_units, self.units * 2),
            nn.ReLU(),
            nn.Linear(self.units * 2, 1),
            nn.Hardsigmoid()
        ]

        self.dmp_dt_model_pre = nn.ModuleList(pre_layers).to(input.device)
        self.dmp_dt_model_post = nn.ModuleList(post_layers).to(input.device)

        def build_model(inpt, use_dropout):
            result = inpt
            for l in self.dmp_dt_model_pre:
                result = l(result)

            if use_dropout:
                result = nn.Dropout(p=0.25)(result)

            for l in self.dmp_dt_model_post:
                result = l(result)
            return result + 0.1

        self.dmp_dt_model = build_model'''



    def forward(self, inputs, training=False, use_dropout=True, node = None, return_cfeature = False):
        if training:
            use_dropout = True

        language_in   = inputs[0]
        #print(f'vor if in model: {language_in}')
        if language_in is not self.last_language:
            self.last_language = language_in

            language_in_tf = tf.convert_to_tensor(language_in.cpu().numpy())
            #language_in_tf = tf.ones_like(language_in_tf)
            language  = self.embedding(language_in_tf)
            #print(f'language after embedding: {language[0, :5,:5]}')

            language = torch.tensor(language.numpy(), device=inputs[1].device)
            if self.lng_gru is None:
                self.build_lang_gru(language)
            self.lng_gru = set_up_GRU_paras_torch(self.lng_gru)
            print(f'langafe shape: {language.shape}')
            language[:,:3] = torch.zeros_like(language[:,:3])
            _, language  = self.lng_gru(language) 
            language = language.squeeze()

        print(f'language after gru: {language[:5,:5]}')
        return
        features   = inputs[1]
        # local      = features[:,:,:5]
        robot      = inputs[2]

        # dmp_state  = inputs[3]


        batch_size = robot.size(0)

        # Calculate attention and expand it to match the feature size
        self.attention = load_Att_Torch(self.attention)
        atn = self.attention((language, features))
        atn_w = atn.unsqueeze(2)
        atn_w = atn_w.repeat([1,1,5])
        # Compress image features and apply attention
        cfeatures = atn_w * features
        cfeatures = cfeatures.sum(axis=1)

        # Add the language to the mix again. Possibly usefull to predict dt
        start_joints  = robot[:,0,:]

        cfeatures = torch.cat((cfeatures, language, start_joints), axis=1)
        #cfeatures = tf.keras.backend.concatenate((cfeatures, language, start_joints), axis=1)

        # Policy Translation: Create weight + goal for DMP
        if self.dmp_dt_model is None:
            self.dmp_dt_model = dmp_dt_torch(ptgloabl_units=self.ptgloabl_units, units=self.units)
            self.dmp_dt_model.build_dmp_dt_model(cfeatures)
        #print(f'cfeatures: {cfeatures.shape}')
        self.dmp_dt_model = load_torch_dmp(self.dmp_dt_model)

        dmp_dt = self.dmp_dt_model.dmp_model(inpt = cfeatures, use_dropout=False)
        #print(f'dmp_dt: {dmp_dt}')
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
        '''print('number of parameters')
        print(f'lng gru: {len(list(self.lng_gru.parameters()))}')

        print(f'attention: {len(list(self.attention.parameters()))}')
        print(f'phase model: {len(list(self.dmp_dt_model_seq.parameters()))}')
        print(f'controller: {len(list(self.controller.parameters()))}')
        print(f'overall {len(list(self.parameters()))}')'''

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
    
    def saveModelToFile(self, add):
        dir_path = path.dirname(path.realpath(__file__))
        path_to_file = dir_path + "/Data/Model/" + add
        if not path.exists(path_to_file):
            makedirs(path_to_file)
        print(f'path to file: {path_to_file}')
        torch.save(self.state_dict(), path_to_file + "policy_translation")