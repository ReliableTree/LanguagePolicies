# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import pickle
from numpy.core.fromnumeric import mean
import tensorflow as tf
import numpy as np
import pathlib

from tensorflow.python.eager.context import device

from model_src.attention import TopDownAttention
from model_src.attentionTorch import TopDownAttentionTorch
from model_src.glove import GloveEmbeddings
from model_src.feedbackcontrollerTorch import FeedBackControllerTorch
from model_src.controllerTransformer import ControllerTransformer
from model_src.transformerAttention import TransformerAttention
from model_src.planNetwork import Plan_NN
from model_src.transformerUpConv import TransformerUpConv

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
    def __init__(self, od_path, glove_path = None, model_setup=None):
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
            self.prediction_nn = None

        self.memory = {}

    def reset_memory(self):
        for name in self.memory:
            self.memory[name] = None

    def load_memory(self, path, names):
        for name in names:
            self.memory[name] = torch.load(path + name, map_location='cuda:0')

    def build_lang_gru(self, input, bias = True):
        self.lng_gru = nn.GRU(input.size(-1), self.units, 1, batch_first = True, device=input.device, bias = bias)


    def forward(self, inputs, gt_attention = None, gt_tjkt=None, mean_over_do = False, optimize = False):
        result = {}
        ###
        if 'meta_world' in self.model_setup and self.model_setup['meta_world']['use']:
            inpt_features = inputs[:,:1].transpose(0,1)
            seq_len = self.model_setup['meta_world']['seq_len']
        else:
            seq_len = 350
            language_in   = inputs[0]
            features   = inputs[1]                          #16x6x5 first entry is object class
            robot      = inputs[2]                           #16x350x7
            if self.model_setup['use_memory']:
                self.use_memory(robot[:,0], 'start_position')

            if (self.last_language is None) or (not torch.equal(language_in, self.last_language)):
                pass   #reactivate if recurrent
            self.language = self.get_language(language_in)               #16 x 32

            # Calculate attention and expand it to match the feature size
            
            inpt_features, diff = self.get_inpt_features(features, gt_attention, robot) #1x16x53
            result['atn']     = self.obj_atn
            result['diff']    = diff
            #inpt_features = inpt_features[:,:1,:].repeat([1, inpt_features.size(1), 1])
        #print(f'inpt featrue: {inpt_features.shape}')
        current_plan = self.get_plan(inpt_features, seq_len) #350x16x8
        #if optimize:
        #    current_plan = self.optimize(current_plan, inpt_features)

        #if 'predictionNN' in self.model_setup['contr_trans'] and self.model_setup['contr_trans']['predictionNN']:
        #    predicted_loss_p, predicted_loss_gt = self.pred_forward(inpt_features, current_plan, gt_tjkt, mean_over_do)
        
        current_plan = current_plan.transpose(0,1) #16x350x8
        result['gen_trj'] = current_plan[:,:,:-1]
        result['phs']     = current_plan[:,:,-1]


        #if 'predictionNN' in self.model_setup['contr_trans'] and self.model_setup['contr_trans']['predictionNN']:
        #    result['loss_prediction'] = predicted_loss_p
        #    result['loss_gt']         = predicted_loss_gt
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

    def optimize(self, inpt, input_features):

        import copy
        opt_inpt = torch.clone(inpt)
        opt_inpt = opt_inpt.detach()
        opt_inpt = torch.zeros_like(opt_inpt)
        print('before')
        print(opt_inpt[0,0])
        #print(opt_inpt)
        opt_inpt.requires_grad = True
        opt = torch.optim.Adam([opt_inpt], lr=1e20)
        epochs = 1000
        self.prediction_nn_copy = copy.deepcopy(self.prediction_nn)
        opt_input_features = torch.zeros_like(input_features)

        for i in range(epochs):
            opt.zero_grad()
            loss = (self.pred_forward(inpt_features=opt_input_features, current_plan=opt_inpt, freeze = False)[0]).sum()
            #print(f'loss in optimizer: {loss}')
            loss.backward()
            opt.step()
            if i%100==0:
                #print(opt_inpt.shape)
                print(opt_inpt._grad[0,0])
                print(opt_inpt[0,0])
                print(f'loss in optimizer: {loss}')
        return opt_inpt

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
        '''if self.model_setup['use_memory']:
            if self.model_setup['train']:
                if self.mem is None:
                    self.mem = language
                else:
                    self.mem = torch.cat((self.mem, language))
            else:
                print('load')
                language = self.find_closest_match(language)'''
        
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

    def get_plan(self, inpt_features, seq_len):
        if self.plan_nn is None:
            num_upconvs = self.model_setup['contr_trans']['plan_nn']['plan']['num_upconvs']
            stride= self.model_setup['contr_trans']['plan_nn']['plan']['stride']
            d_output= self.model_setup['contr_trans']['plan_nn']['plan']['d_output']
            nhead= self.model_setup['contr_trans']['plan_nn']['plan']['nhead']
            d_hid= self.model_setup['contr_trans']['plan_nn']['plan']['d_hid']
            nlayers= self.model_setup['contr_trans']['plan_nn']['plan']['nlayers']
            use_layernorm = self.model_setup['contr_trans']['plan_nn']['plan']['use_layernorm']
            self.plan_nn = TransformerUpConv(num_upconvs=num_upconvs, stride=stride, ntoken=inpt_features.size(-1), d_output=d_output, d_model=d_hid, nhead=nhead, d_hid=d_hid, nlayers=nlayers, seq_len=seq_len, use_layernorm=use_layernorm).to(inpt_features.device)
        return self.plan_nn.forward(inpt_features) #350x16x8
    
    def pred_forward(self, inpt_features, current_plan, gt_tjkt=None, mean_over_do=False, freeze = False):
        import copy
        if mean_over_do:
            c_plan = current_plan.mean(dim=1).unsqueeze(1).repeat([1,current_plan.size(1),1]) #350 x 16 x 53
        else:
            c_plan = current_plan
        pred_features = inpt_features.repeat([c_plan.size(0), 1, 1]) #350 x 16 x 53
        pred_features_p = torch.cat((pred_features, c_plan[:,:,:7]), dim = -1) #350x16x60
        if self.prediction_nn is None:
            num_upconvs = self.model_setup['contr_trans']['plan_nn']['plan']['num_upconvs']
            stride= self.model_setup['contr_trans']['plan_nn']['plan']['stride']
            nhead= self.model_setup['contr_trans']['plan_nn']['plan']['nhead']
            d_hid= self.model_setup['contr_trans']['plan_nn']['plan']['d_hid']
            nlayers= self.model_setup['contr_trans']['plan_nn']['plan']['nlayers']
            use_layernorm = self.model_setup['contr_trans']['plan_nn']['plan']['use_layernorm']
            self.prediction_nn = TransformerUpConv(num_upconvs=num_upconvs, stride=stride, ntoken=pred_features_p.size(-1), d_output=1, d_model=d_hid, nhead=nhead, d_hid=d_hid, nlayers=nlayers, seq_len=350, use_layernorm=use_layernorm, upconv=False).to(current_plan.device)

        if freeze:
            predicted_loss_p = self.prediction_nn_copy(pred_features_p).squeeze() #1x16x1
        else:
            predicted_loss_p = self.prediction_nn(pred_features_p).squeeze() #1x16x1
        if gt_tjkt is not None:
            pred_features_gt = torch.cat((pred_features, gt_tjkt.transpose(0,1)), dim=-1) #350x16x60
            predicted_loss_gt = self.prediction_nn(pred_features_gt).squeeze()
        else:
            predicted_loss_gt = None
        return predicted_loss_p, predicted_loss_gt

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