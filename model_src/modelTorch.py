# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import pickle
from readline import set_completion_display_matches_hook
from numpy.core.fromnumeric import mean
import tensorflow as tf
import pathlib

from LanguagePolicies.model_src.attentionTorch import TopDownAttentionTorch
from LanguagePolicies.model_src.glove import GloveEmbeddings
from LanguagePolicies.model_src.transformerAttention import TransformerAttention

from LanguagePolicies.utils.Transformer import TransformerModel
from LanguagePolicies.utils.Transformer import generate_square_subsequent_mask
from MetaWorld.utilsMW.model_setup_obj import ModelSetup

import torch
import torch.nn as nn

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
    def __init__(self, od_path, glove_path = None, model_setup:ModelSetup = None, device = 'cpu'):
        super().__init__()
        self.device = device
        self.model_setup         = model_setup

        self.embedding = GloveEmbeddings(file_path=glove_path)
        self.dout      = nn.Dropout(p=0.25)

        self.plan_nn = None

        self.memory = {}

    def reset_memory(self):
        for name in self.memory:
            self.memory[name] = None

    def load_memory(self, path, names):
        for name in names:
            self.memory[name] = torch.load(path + name, map_location='cuda:0')

    def forward(self, inputs):
        #print(f'inptuts: {inputs[0,:3]}')
        result = {}
        #inpt_features = inputs[:,:1].transpose(0,1)
        inpt_features = inputs.transpose(0,1)
        
        #print(f'inpt: {inpt_features.shape}')
        current_plan = self.get_plan(inpt_features) #350x16x8
        current_plan = current_plan.transpose(0,1) #16x350x8
        result['gen_trj'] = current_plan
        result['inpt_trj'] = current_plan
        return result


    def find_closest_match(self, name, inpt, robot):
        if robot is not None:
            pickup, comp_vec = self.is_pickup(robot)
            dist = ((self.memory['start_position']-comp_vec)**2).sum(dim = -1)
            if pickup:
                in_memory = self.memory[name][dist < 0.01]
            else:
                in_memory = self.memory[name][dist > 0.01]
        else:
            in_memory = self.memory[name]
        diff = ((inpt.unsqueeze(0) - in_memory.unsqueeze(1))**2).sum(dim=-1)
        best_match = torch.argmin(diff, dim=0)
        return in_memory[best_match], torch.min(diff)

    def is_pickup(self, robot):
        comp_vec = torch.tensor([0.4499456882, 0.2921932340, 0.5975010395, 0.9999295473, 0.8829264045,
        0.4917714894, 0.0000000000]).to(robot.device)
        return ((comp_vec - robot[0])**2).sum() < 1, comp_vec


    def get_language(self, language_in):
        self.last_language = language_in

        language_in_tf = tf.convert_to_tensor(language_in.cpu().numpy())
        language  = self.embedding(language_in_tf)

        language = torch.tensor(language.numpy(), device=language_in.device)
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
        if 'bottleneck' in self.model_setup['lang_trans'] and self.model_setup['lang_trans']['bottleneck']:
            language = nn.Softmax(dim=-1)(language.squeeze())

        language = language.squeeze()
        
        return language.squeeze() 

    def get_max_features(self, features, gt_attention = None):
        if self.model_setup['obj_embedding']['use_obj_embedding']:
            if self.obj_embedding is None:
                eis = self.model_setup['obj_embedding']['EIS'] #30
                eos = self.model_setup['obj_embedding']['EOS'] #10
                self.obj_embedding = nn.Embedding(eis, eos).to(features.device)
            if self.model_setup['obj_embedding']['train_embedding']:
                obj_feature_embedding = self.obj_embedding(features[:,:,0].to(dtype=torch.int32))   #16x10
            else:
                with torch.no_grad():
                    obj_feature_embedding = self.obj_embedding(features[:,:,0].to(dtype=torch.int32))
            obj_feature_embedding = torch.cat((features[:,:,1:], obj_feature_embedding), dim = -1)
        else:
            obj_feature_embedding = features

        if self.attention is None and self.model_setup['attn_trans']['use_attn_trans']:
            self.attention = TransformerAttention(device = features.device)
        self.obj_atn = self.attention((self.language, obj_feature_embedding))
        if gt_attention is None:
            main_obj = torch.argmax(self.obj_atn, dim=-1)
        else:
            main_obj = torch.argmax(gt_attention, dim=-1) 
        #print(f'atn.shape, {atn.shape}')

        counter = torch.arange(features.size(0))
        return obj_feature_embedding[(counter, main_obj)]

    def use_memory(self, vector, name, robot = None):
        diff = 0
        if self.model_setup['train']:
            if (name not in self.memory) or (self.memory[name] is None):
                self.memory[name] = vector
            else:
                self.memory[name] = torch.cat((self.memory[name], vector))
            result = vector
        else:
            result, diff = self.find_closest_match(name, vector, robot)
            if name == 'cfeatures':
                result[:,:4] = vector[:,:4]
        return result, diff

    def get_inpt_features(self, features, gt_attention, robot):
        features_max = self.get_max_features(features, gt_attention)          #16x5
        cfeatures = torch.cat((features_max, self.language), axis=1)   #16x46
        diff = 0
        if self.model_setup['use_memory']:
            cfeatures, diff = self.use_memory(cfeatures, 'cfeatures', robot[:,0])
        return torch.cat((cfeatures, robot[:,0]), dim=-1).unsqueeze(0), diff  #1x16x46+7 = 1x16x53

    def smoothing(self, inpt):
        # Create gaussian kernels
        shape = inpt.shape
        if self.kernel is None:
            self.kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]).to(inpt.device)
        # Apply smoothing
        smoothed =  torch.nn.functional.conv1d(inpt.reshape(1,1,-1), self.kernel, padding='same')
        return smoothed.reshape([*shape])

    def get_plan(self, inpt_features):
        #in_transformer = inpt_features.repeat(self.model_setup.seq_len, 1, 1)
        in_transformer = inpt_features
        if (self.plan_nn is None):
            model_setup = self.model_setup
            model_setup.ntoken = inpt_features.size(-1)
            self.plan_nn = TransformerModel(model_setup=self.model_setup).to(inpt_features.device)
        plan = self.plan_nn.forward(in_transformer) #350x16x8

        return plan
    

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
        for name in self.memory:
            torch.save(self.memory[name], path_to_file + name)