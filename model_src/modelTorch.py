# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import pickle
import tensorflow as tf
import numpy as np
import pathlib

from torch._C import device, dtype
from model_src.attention import TopDownAttention
from model_src.attentionTorch import TopDownAttentionTorch
from model_src.glove import GloveEmbeddings
from model_src.feedbackcontrollerTorch import FeedBackControllerTorch
from model_src.controllerTransformer import ControllerTransformer
from model_src.transformerAttention import TransformerAttention
from model_src.planNetwork import Plan_NN

from utils.Transformer import TransformerModel
from utils.Transformer import generate_square_subsequent_mask
from utils.torch_util import dmp_dt_torch

import torch
import torch.nn as nn
import time

from os import path, makedirs

#model_setup: {obj_embedding:{use_obj_embedding, EIS, EOS} , attn_trans{use_attn_trans, }, lang_trans"{use_lang_trans,  }, contr_trans:{use_contr_trans, }, LSTM:{use_LSTM} }

'''lang_trans:                    d_output = self.model_setup['lang_trans']['d_output'] #32
                                d_model = self.model_setup['lang_trans']['d_model']   #42
                                nhead = self.model_setup['lang_trans']['nhead']   #2
                                nlayers = self.model_setup['lang_trans']['nlayers']   #2'''

'''obj_embedding : EIS:30
                EOS:10'''

'''
contr_trans:                 d_output = self.model_setup['contr_trans']['d_output'] #8
                            d_model = self.model_setup['contr_trans']['d_model'] #210
                            nhead   = self.model_setup['contr_trans']['nhead'] #6
                            nlayers = self.model_setup['contr_trans']['nlayers'] #4
'''

class PolicyTranslationModelTorch(nn.Module):
    def __init__(self, od_path, glove_path, model_setup):
        super().__init__()
        self.model_setup         = model_setup
        self.units               = 32
        self.output_dims         = 7
        self.basis_functions     = 11
        self.ptgloabl_units      = 42

        if od_path != "":                
            od_path    = pathlib.Path(od_path)/"saved_model" 
            self.frcnn = tf.saved_model.load(str(od_path))
            self.frcnn = self.frcnn.signatures['serving_default']
            self.frcnn.trainable = False

        if model_setup['obj_embedding']['use_obj_embedding']:
            self.obj_embedding = None
        if model_setup['attn_trans']['use_attn_trans']:
            self.attention = None
        else:
            self.attention = TopDownAttentionTorch(units=64)


        self.embedding = GloveEmbeddings(file_path=glove_path)
        if model_setup['lang_trans']['use_lang_trans']:
            self.transformer_seq_embedding = None
            self.lang_trans  = None
        else:
            self.lng_gru   = None


        self.dout      = nn.Dropout(p=0.25)

        # Units needs to be divisible by 7
        if model_setup['contr_trans']['use_contr_trans']:
            self.controller_transformer = None
            count_emb_dim = model_setup['contr_trans']['count_emb_dim']
            self.count_embedding = nn.Embedding(350, count_emb_dim)
        else:
            self.dmp_dt_model = None
            self.controller = FeedBackControllerTorch(
                robot_state_size=self.units,
                dimensions=self.output_dims,
                basis_functions=self.basis_functions,
                cnfeatures_size=self.units + self.output_dims + 5,
                use_LSTM = model_setup['LSTM']['use_LSTM'] 
            )

        self.last_language = None

        if 'plan_nn' in self.model_setup['contr_trans'] and self.model_setup['contr_trans']['plan_nn']['use_plan_nn']:
            self.plan_nn = None



    def build_lang_gru(self, input, bias = True):
        self.lng_gru = nn.GRU(input.size(-1), self.units, 1, batch_first = True, device=input.device, bias = bias)


    def forward(self, inputs, training=False, return_cfeature = False, gt_attention = None, model_params = None):
        if training:
            gt_attention = None

        language_in   = inputs[0]
        #print(f'vor if in model: {language_in}')
        if (self.last_language is None) or (not torch.equal(language_in, self.last_language)):
            self.last_language = language_in

            language_in_tf = tf.convert_to_tensor(language_in.cpu().numpy())
            language  = self.embedding(language_in_tf)

            language = torch.tensor(language.numpy(), device=inputs[1].device)
            if self.model_setup['lang_trans']['use_lang_trans']:
                if self.lang_trans is None:
                    d_output = self.model_setup['lang_trans']['d_output'] #32
                    d_model = self.model_setup['lang_trans']['d_model']   #42
                    nhead = self.model_setup['lang_trans']['nhead']   #2
                    nlayers = self.model_setup['lang_trans']['nlayers']   #2

                    self.lang_trans = TransformerModel(ntoken=language.size(-1), d_output=d_output, d_model=d_model, nhead=nhead, d_hid=d_model, nlayers=nlayers)
                    self.lang_trans.to(language.device)
                language = self.lang_trans(language.transpose(0,1)) #size: S, N, D
                #print(f'language after trans: {language.shape}')
                language = language.transpose(0,1)                  #size: N, S, D
                #print(f'language after transpose: {language.shape}')
                if self.transformer_seq_embedding is None:
                    self.transformer_seq_embedding = nn.Linear(language.size(-2) * language.size(-1), language.size(-1)).to(language.device)
                language = self.transformer_seq_embedding(language.reshape(language.size(0), -1))
                #print(f'language after embedding: {language.shape}')
            else:
                if self.lng_gru is None:
                    self.build_lang_gru(language)
                _, language  = self.lng_gru(language) 
            #print(f'language shape: {language.shape}')
            self.language = language.squeeze()               #16 x 32

            features   = inputs[1]                          #16x6x5 first entry is object class
            # Calculate attention and expand it to match the feature size
            if self.model_setup['obj_embedding']['use_obj_embedding']:
                if self.obj_embedding is None:
                    eis = self.model_setup['obj_embedding']['EIS'] #30
                    eos = self.model_setup['obj_embedding']['EOS'] #10
                    self.obj_embedding = nn.Embedding(eis, eos).to(features.device)
                if model_params['obj_embedding']['train_embedding']:
                    obj_feature_embedding = self.obj_embedding(features[:,:,0].to(dtype=torch.int32))   #16x10
                else:
                    with torch.no_grad():
                        obj_feature_embedding = self.obj_embedding(features[:,:,0].to(dtype=torch.int32))
                obj_feature_embedding = torch.cat((features[:,:,1:], obj_feature_embedding), dim = -1)
            else:
                obj_feature_embedding = features
            if self.attention is None and self.model_setup['attn_trans']['use_attn_trans']:
                self.attention = TransformerAttention(device = features.device)
            self.atn = self.attention((self.language, obj_feature_embedding))
            if gt_attention is None:
                main_obj = torch.argmax(self.atn, dim=-1)
            else:
                main_obj = torch.argmax(gt_attention, dim=-1) 
            #print(f'atn.shape, {atn.shape}')

            counter = torch.arange(features.size(0))
            self.cfeatures_max = obj_feature_embedding[(counter, main_obj)]           #16x5

        robot      = inputs[2]                           #16x350x7
        robot_first_state = robot[:,:1].repeat(1, 350,1) #16x350x7                    #
        counter = torch.arange(350, device=robot.device)
        if model_params['contr_trans']['use_counter_embedding']:
            if model_params['obj_embedding']['train_embedding']:
                ce = self.count_embedding(counter).unsqueeze(0).repeat((robot.size(0), 1, 1)) #16x350x20
            else:
                with torch.no_grad():
                    ce = self.count_embedding(counter).unsqueeze(0).repeat((robot.size(0), 1, 1)) #16x350x20
        else:
            counter = counter/200
            ce = counter.reshape(1,-1,1)
            
            ce = ce.repeat((robot.size(0), 1, 1))
        robot_first_state = torch.cat((robot_first_state, ce), dim=-1)

        # Add the language to the mix again. Possibly usefull to predict dt

        if self.model_setup['contr_trans']['use_contr_trans'] :
            cfeatures = torch.cat((self.cfeatures_max, self.language), axis=1)   #16x46
            if 'plan_nn' in self.model_setup['contr_trans'] and self.model_setup['contr_trans']['plan_nn']['use_plan_nn']:
                inpt_features = torch.cat((cfeatures, robot[:,0]), dim=-1) #16x46+7 = 16x53
                if self.plan_nn is None:
                    self.plan_nn = Plan_NN(inpt_features.size(-1), self.model_setup['contr_trans']['plan_nn']['plan_outpt_dim'] * 350).to(robot.device)
                inpt_seq = self.plan_nn.forward(inpt_features)
                inpt_seq = inpt_seq.reshape(robot.size(0), 350, -1)

            else:
                cfeatures = cfeatures.unsqueeze(1).repeat(1,robot_first_state.size(1),1)         #16x350x46
                inpt_seq = torch.cat((cfeatures, robot_first_state), dim = -1)          #16x350x53 + 20

            if self.controller_transformer is None:
                d_output = self.model_setup['contr_trans']['d_output'] #8
                d_model = self.model_setup['contr_trans']['d_model'] #210
                nhead   = self.model_setup['contr_trans']['nhead'] #6
                nlayers = self.model_setup['contr_trans']['nlayers'] #4
                self.controller_transformer = ControllerTransformer(ntoken=inpt_seq.size(-1), d_output=d_output, d_model=d_model, nhead=nhead, d_hid=d_model, nlayers=nlayers).to(inpt_seq.device)
                self.trans_seq_len = max(inpt_seq.size(1), 350)

            src_mask = generate_square_subsequent_mask(inpt_seq.size(1)).to(inpt_seq.device)

            #result from gt:
            if model_params['contr_trans']['recursive']:
                generated_recursive = inpt_seq[:,:1].transpose(0,1) #1x16x53
                for step in range(inpt_seq.size(1) - 1):
                    if step < (inpt_seq.size(1) - 2):
                        with torch.no_grad():
                            res = self.controller_transformer(generated_recursive, src_mask = src_mask[:(step+1), :(step + 1)])[-1].unsqueeze(0) #1x16x8
                    else:
                        res = self.controller_transformer(generated_recursive, src_mask = src_mask[:(step+1), :(step + 1)])[-1].unsqueeze(0) #1x16x8
                    n_state = torch.cat((cfeatures[:,:1].transpose(0,1), res[:,:,:7]), dim = -1)
                    generated_recursive = torch.cat((generated_recursive, n_state), dim=0) 
                generated_from_gt = generated_recursive.transpose(0,1)
            else:
                if model_params['contr_trans']['use_mask']:
                    generated_from_gt = self.controller_transformer(inpt_seq.transpose(0,1), src_mask = src_mask)  #350x16x8
                else:
                    generated_from_gt = self.controller_transformer(inpt_seq.transpose(0,1), src_mask=None)
                generated_from_gt = generated_from_gt.transpose(0,1)                              #16x350x8

            #result from generated:
            if model_params['contr_trans']['use_gen2']:
                with torch.no_grad():
                    generated_for_generation = self.controller_transformer(inpt_seq.transpose(0,1), src_mask = src_mask).transpose(0,1)      #350x16x8
                generated_trj_for_generation = generated_for_generation[:,:,:7]
                inpt_seq_generated = torch.cat((cfeatures, generated_trj_for_generation), dim = -1)
                generated_from_generated = self.controller_transformer(inpt_seq_generated.transpose(0,1), src_mask = src_mask).transpose(0,1) 
                return generated_from_gt[:,:,:7], generated_from_generated[:,:,:7], self.atn, generated_from_generated[:,:,7]
            else:
                return generated_from_gt[:,:,:7], self.atn, generated_from_gt[:,:,7]

    
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
        torch.save(self.state_dict(), path_to_file + "policy_translation_h")
        with open(path_to_file + 'model_setup.pkl', 'wb') as f:
            pickle.dump(self.model_setup, f)  
