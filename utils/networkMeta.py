# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals
from os import name, path
from pickle import NONE
from pickletools import read_uint1
from tabnanny import verbose
from tkinter.messagebox import NO
import tensorflow as tf
import sys
import numpy as np
from utils.graphsTorch import TBoardGraphsTorch
from utilsMW.metaOptimizer import SignalModule, TaylorSignalModule, meta_optimizer, tailor_optimizer
from utilsMW.makeTrainingData import SuccessSimulation
from utilsMW.dataLoaderMW import TorchDatasetTailor
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import time
import copy

class NetworkMeta(nn.Module):
    def __init__(self, model, tailor_models, env_tag, data_path, logname, lr, lw_atn, lw_w, lw_trj, lw_gen_trj, lw_dt, lw_phs, lw_fod, log_freq=25, gamma_sl = 0.995, device = 'cuda', use_transformer = True, tboard=True):
        super().__init__()
        self.optimizer         = None
        self.model             = model
        self.tailor_models      = tailor_models
        self.total_steps       = 0
        self.logname           = logname
        self.lr = lr
        self.device = device
        self.data_path = data_path
        self.use_transformer = use_transformer
        self.use_tboard = tboard
        self.embedding_memory = {}
        self.env_tag = env_tag
        self.init_train = True
        self.model_state_dict = self.model.state_dict()
        self.max_success_rate = 0

        if self.logname.startswith("Intel$"):
            self.instance_name = self.logname.split("$")[1]
            self.logname       = self.logname.split("$")[0]
        else:
            self.instance_name = None

        if tboard:
            self.tboard            = TBoardGraphsTorch(self.logname, data_path=data_path)
        self.loss              = nn.CrossEntropyLoss()
        self.global_best_loss  = float('inf')
        self.global_best_loss_val = float('inf')
        self.last_written_step = -1
        self.log_freq          = log_freq

        self.lw_atn = lw_atn 
        self.lw_w   = lw_w 
        self.lw_trj = lw_trj
        self.lw_dt  = lw_dt
        self.lw_phs = lw_phs
        self.lw_fod = lw_fod
        self.lw_gen_trj = lw_gen_trj

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.gamma_sl = gamma_sl

        self.successSimulation = SuccessSimulation()
        self.trajectories = None
        self.inpt_obs= None
        self.success= None

    def setup_model(self, model_params):
        with torch.no_grad():
            for step, (d_in, d_out) in enumerate(self.train_ds):
                result = self.model(inputs=d_in)
                break
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, 0.9, verbose=True)
        #self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr) 

        self.signal_main = SignalModule(model=self.model, loss_fct=self.calculateLoss, optimizer=self.optimizer)

        trajectories, inpt_obs, success = self.successSimulation.get_success(policy = self.model, env_tag = self.env_tag, n = 2)
        inpt_obs = inpt_obs.repeat((1, trajectories.size(1), 1))
        inpt = torch.concat((trajectories, inpt_obs), dim = -1)
        #inpt = inpt.transpose(0,1)
        with torch.no_grad():
            for tmodel in self.tailor_models:
                t_result = tmodel.forward(inpt)
        self.tailor_optimizers = [torch.optim.Adam(params=tailor_model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0) for tailor_model in self.tailor_models]
        self.meta_optimizers = [torch.optim.Adam(params=tailor_model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0) for tailor_model in self.tailor_models]
        #self.tailor_optimizers = [torch.optim.SGD(params=tailor_model.parameters(), lr=0.05*self.lr) for tailor_model in self.tailor_models]
        #self.meta_optimizers = [torch.optim.SGD(params=tailor_model.parameters(), lr=self.lr) for tailor_model in self.tailor_models]
        
        
        def lfp(result, label):
            return ((result-label)**2).mean()
        self.tailor_modules = []
        for i in range(len(self.tailor_models)):
            self.tailor_modules.append(TaylorSignalModule(model=self.tailor_models[i], loss_fct=lfp, optimizer=self.tailor_optimizers[i], meta_optimizer=self.meta_optimizers[i]))
        

    def setDatasets(self, train_loader, val_loader):
        self.train_ds = train_loader
        self.val_ds   = val_loader


    def train_tailor(self, ):
        trajectories, inpt_obs, success = self.successSimulation(policy = self.model, env_tag = self.env_tag, n = 10)
        inpt = torch.concat((trajectories, inpt_obs), dim = 0)
        debug_dict = tailor_optimizer(tailor_modules=self.tailor_modules, inpt=inpt, label=success)
        self.write_tboard_train_scalar(debug_dict=debug_dict)


    def train(self, epochs, model_params):
        self.global_step = 0
        for epoch in range(epochs):
            if (epoch) % model_params['val_every'] == 0:
                print(f'logname: {self.logname}')
                self.runValidation(quick=False, epoch=epoch, save=True, model_params=model_params)

            self.model.reset_memory()
            rel_epoch = epoch/epochs
            print("Epoch: {:3d}/{:3d}".format(epoch+1, epochs)) 
            validation_loss = 0.0
            train_loss = []
            model_params['obj_embedding']['train_embedding']  = rel_epoch < 1.1
            #print(f'train embedding: {model_params["obj_embedding"]["train_embedding"]}')
            self.model.model_setup['train'] = True
            if not self.init_train:
                loss_module = 1

                while loss_module > 0.01:
                    lma = None
                    for trajectories, inpt_obs, success in self.tailor_loader:
                        self.global_step += 1
                        debug_dict = self.tailor_step(trajectories, inpt_obs, success)
                        if 'tailor_module' in debug_dict:
                            if lma is None:
                                lma = debug_dict['tailor_module'].reshape(1)
                            else:
                                lma = torch.cat((lma, debug_dict['tailor_module'].reshape(1)), 0)
                            loss_module = lma.mean()
            for step, (d_in, d_out) in enumerate(self.train_ds):
                #print(f'inpt shape: {d_in.shape}')
                if (step+1) % 400 == 0:
                    validation_loss = self.runValidation(quick=True, pnt=False, epoch=epoch, save = False, model_params=model_params)   
                    self.model.model_setup['train'] = True     
                train_loss.append(self.step(d_in, d_out, train=True, model_params=model_params))
                self.loadingBar(step, self.total_steps, 25, addition="Loss: {:.6f} | {:.6f}".format(np.mean(train_loss[-10:]), validation_loss))
                if epoch == 0:
                    self.total_steps += 1
                self.global_step += 1

            self.loadingBar(self.total_steps, self.total_steps, 25, addition="Loss: {:.6f}".format(np.mean(train_loss)), end=True)
            #self.train_tailor()
            


            if self.use_tboard:
                self.model.saveModelToFile(add = self.logname + "/", data_path = self.data_path)
            self.scheduler.step()

    
    
    def tailor_step(self, trajectories, inpt_obs, success):
        debug_dict = tailor_optimizer(tailor_modules = self.tailor_modules, trajectories = trajectories, inpt=inpt_obs, label=success)
        debug_dict['rel_succ'] = success.mean()
        self.write_tboard_scalar(debug_dict=debug_dict, train=True)
        return debug_dict
    
    def torch2tf(self, inpt):
        if inpt is not None:
            return tf.convert_to_tensor(inpt.detach().cpu().numpy())
        else:
            return None
    def tf2torch(self, inpt):
        if inpt is not None:
            return torch.tensor(inpt.numpy(), device= self.device)
        else:
            return None
    def runValidation(self, quick=False, pnt=True, epoch = 0, save = False, model_params = {}): 
        model_params = copy.deepcopy(model_params) #dont change model params globally
        self.model.model_setup['train'] = False
        #with torch.no_grad():
        if (not quick):
            print("Running full validation...")
            trajectories, inpt_obs, success = self.successSimulation.get_success(policy = self.model, env_tag = self.env_tag, n=150)
            debug_dict = {'success rate' : success.mean()}
            self.write_tboard_scalar(debug_dict=debug_dict, train=False)

            mean_success = success.mean()
            if mean_success > self.max_success_rate:
                self.max_success_rate = mean_success
            else:
                self.model.load_state_dict(self.model_state_dict)

            if mean_success > 0.5 or True:
                self.init_train = False
                if self.trajectories is None:
                    self.trajectories = trajectories
                    self.inpt_obs = inpt_obs
                    self.success = success
                else:
                    self.trajectories = torch.cat((self.trajectories, trajectories), dim=0)[:-12000]
                    self.inpt_obs = torch.cat((self.inpt_obs, inpt_obs), dim=0)[:-12000]
                    self.success = torch.cat((self.success, success), dim=0)[:-12000]
                tailor_data = TorchDatasetTailor(trajectories= self.trajectories, obsv=self.inpt_obs, success=self.success)

                #train_data = torch.utils.data.Subset(train_data, train_indices).to(device)
                self.tailor_loader = DataLoader(tailor_data, batch_size=20, shuffle=True)
            else:
                self.init_train = True
        



        val_loss = []
        for step, (d_in, d_out) in enumerate(self.val_ds):
            loss = self.step(d_in, d_out, train=False, model_params=model_params)
            val_loss.append(loss)
            break
            
            if quick or model_params['quick_val']:
                break

        if self.use_tboard:
            do_dim = d_in[0].size(0)
            #self.model.eval()
            if 'meta_world' in self.model.model_setup and self.model.model_setup['meta_world']['use']:
                out_model = self.model(d_in[:1])
                self.createGraphsMW(1, (d_out[0][0], d_out[1][0]), out_model)
            else:
                in0 = self.tf2torch(tf.tile(tf.expand_dims(self.torch2tf(d_in[0][0]), 0),[do_dim,1]))
                in1 = self.tf2torch(tf.tile(tf.expand_dims(self.torch2tf(d_in[1][0]), 0),[do_dim,1,1]))
                in2 = self.tf2torch(tf.tile(tf.expand_dims(self.torch2tf(d_in[2][0]), 0),[do_dim,1,1]))
                d_in_graphs  = (in0, in1, in2)
                out_model = self.model(d_in_graphs)
                self.createGraphs((d_in[0][0], d_in[1][0], d_in[2][0]),
                                (d_out[0][0], d_out[1][0], d_out[2][0], d_out[3][0]), 
                                out_model,
                                save=save,
                                name_plot = str(step),
                                epoch=epoch,
                                model_params=model_params)

            #self.model.train()


        #self.model.eval()
        #out_model_tf = [self.torch2tf(out_model[0]), [self.torch2tf(ele) for ele in out_model[1]]]
        #d_out_graphs = (tf.tile(tf.expand_dims(d_out[0][0], 0),[50,1,1]), tf.tile(tf.expand_dims(d_out[1][0], 0),[50,1]), 
        #                tf.tile(tf.expand_dims([d_out[2][0]], 0),[50,1]), tf.tile(tf.expand_dims(d_out[3][0], 0),[50,1,1]))

        loss = np.mean(val_loss)
        if pnt:
            print("  Validation Loss: {:.6f}".format(loss))
        if not quick:
            if loss < self.global_best_loss_val:
                self.global_best_loss_val = loss
                if self.use_tboard:
                    self.model.saveModelToFile(add=self.logname + "/best_val/", data_path= self.data_path)
                print(f'best val model saved with: {loss}')
        return np.mean(val_loss)

    def write_tboard_scalar(self, debug_dict, train):
        if self.use_tboard:
                for para, value in debug_dict.items():
                    if train:
                        self.tboard.addTrainScalar(para, value, self.global_step)
                    else:
                        self.tboard.addValidationScalar(para, value, self.global_step)



    def step(self, d_in, d_out, train, model_params):

        if train:
            if self.init_train:
                result = self.model(d_in)
                loss, debug_dict = self.calculateLoss(d_out, result, model_params)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            else:
                self.signal_main, _, result, debug_dict = meta_optimizer(
                    main_module = self.signal_main, 
                    tailor_modules = self.tailor_modules, 
                    inpt=d_in, d_out=d_out, 
                    epoch=1,
                    debug_second = False,
                    force_tailor_improvement = False,
                    model_params = model_params)

                loss = debug_dict['main_loss']

            self.write_tboard_scalar(debug_dict=debug_dict, train=True)
        else:
            with torch.no_grad():
                result = self.model(d_in)
                loss, debug_dict = self.calculateLoss(d_out=d_out, result=result, model_params=model_params)
            

            if self.last_written_step != self.global_step:
                if self.use_tboard:
                    self.last_written_step = self.global_step
                    self.write_tboard_scalar(debug_dict=debug_dict, train=False)

                loss = loss.detach().cpu()
                if loss < self.global_best_loss:
                    self.global_best_loss = loss
                    if self.use_tboard:
                        self.model.saveModelToFile(add=self.logname + "/best/", data_path= self.data_path)
                
                    #print(f'model saved with loss: {loss}')

        return loss.detach().cpu().numpy()
    
    def interpolateTrajectory(self, trj, target):
        batch_size     = trj.shape[0]
        current_length = trj.shape[1]
        dimensions     = trj.shape[2]
        result         = np.zeros((batch_size, target, dimensions), dtype=np.float32)
    
        for b in range(batch_size):
            for i in range(dimensions):
                result[b,:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[b,:,i])
        
        return result

    def calculateMSEWithPaddingMask(self, y_true, y_pred, mask):
        mse = (y_true - y_pred)**2
        mse = mse * mask
        return mse.mean(-1)

    def catCrossEntrLoss(self, y_labels, y_pred):
        y_labels_args = torch.argmax(y_labels, dim = -1)
        return nn.NLLLoss()(torch.log(y_pred), y_labels_args)

    def calcMSE(self, a, b):
        return ((a.squeeze() - b.squeeze())**2).mean()

    def calculateLoss(self, d_out, result, model_params, prefix = ''):
        if 'meta_world' in self.model.model_setup and self.model.model_setup['meta_world']['use']:
            generated, phase = d_out
        else:
            generated, attention, delta_t, weights, phase, loss_atn = d_out
            atn = result['atn']
            atn_loss = self.catCrossEntrLoss(attention, atn)
            rel_correct_objects = (torch.argmax(atn, dim=-1) == torch.argmax(attention, dim=-1)).sum()/len(atn)


        gen_trj = result['gen_trj']
        phs = result['phs']

        phs_loss = self.calcMSE(phase, phs)
        trj_loss = self.calcMSE(generated, gen_trj)

        debug_dict = {
            prefix + 'trj_loss':trj_loss,
            prefix + 'phs_loss':phs_loss,
            }
        if 'meta_world' in self.model.model_setup and self.model.model_setup['meta_world']['use']:
            loss = trj_loss * self.lw_trj +\
                phs_loss * self.lw_phs
            debug_dict[prefix + 'main_loss'] = loss
        
        else:
            loss = atn_loss * self.lw_atn + \
                    trj_loss * self.lw_trj + \
                    phs_loss * self.lw_phs
                    #self.lw_fod * fod_loss
            debug_dict = {
                'atn_loss':atn_loss,
                'rel_correct_objects':rel_correct_objects}
        return (loss, debug_dict)
    
    def loadingBar(self, count, total, size, addition="", end=False):
        if total == 0:
            percent = 0
        else:
            percent = float(count) / float(total)
        full = int(percent * size)
        fill = size - full
        print("\r  {:5d}/{:5d} [".format(count, total) + "#" * full + " " * fill + "] " + addition, end="")
        if end:
            print("")
        sys.stdout.flush()

    def createGraphs(self, d_in, d_out, result, save = False, name_plot = '', epoch = 0, model_params={}):
        language, image, robot_states            = d_in
        target_trj, attention, delta_t, weights  = d_out
        gen_trj = result['gen_trj']
        atn     = result['atn']
        phase   = result['phs']


        self.tboard.plotClassAccuracy(attention, self.tf2torch(tf.math.reduce_mean(self.torch2tf(atn), axis=0)), self.tf2torch(tf.math.reduce_std(self.torch2tf(atn), axis=0)), self.tf2torch(self.torch2tf(language)), stepid=self.global_step)
        path_to_plots = self.data_path + "/plots/"+ str(self.logname) + '/' + str(epoch) + '/'
        gen_tr_trj= self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_trj), axis=0))
        gen_tr_phase = self.tf2torch(tf.math.reduce_mean(self.torch2tf(phase), axis=0))
        self.tboard.plotDMPTrajectory(target_trj, gen_tr_trj, torch.zeros_like(gen_tr_trj),
                                    gen_tr_phase, delta_t, None, stepid=self.global_step, save=save, name_plot=name_plot, path=path_to_plots)
        
    def createGraphsMW(self, d_in, d_out, result, save = False, name_plot = '', epoch = 0, model_params={}):
        target_trj, target_phase  = d_out
        gen_trj = result['gen_trj'][0]
        phase   = result['phs'][0]
        path_to_plots = self.data_path + "/plots/"+ str(self.logname) + '/' + str(epoch) + '/'
        #gen_tr_trj= self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_trj), axis=0))
        #gen_tr_phase = self.tf2torch(tf.math.reduce_mean(self.torch2tf(phase), axis=0))
        self.tboard.plotDMPTrajectory(target_trj, gen_trj, torch.zeros_like(gen_trj),
                                    phase, target_phase, None, stepid=self.global_step, save=save, name_plot=name_plot, path=path_to_plots)
        

