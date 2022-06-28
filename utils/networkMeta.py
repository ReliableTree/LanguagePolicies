# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals
from os import name, path, makedirs
from random import triangular
import tensorflow as tf
import sys
import numpy as np
from LanguagePolicies.utils.graphsTorch import TBoardGraphsTorch
from MetaWorld.utilsMW.metaOptimizer import SignalModule, TaylorSignalModule, meta_optimizer, tailor_optimizer, MetaModule
from MetaWorld.utilsMW.dataLoaderMW import TorchDatasetTailor
from LanguagePolicies.utils.Transformer import TailorTransformer
from MasterProject.utils.utils import ConvergenceDetector
from LanguagePolicies.model_src.modelTorch import PolicyTranslationModelTorch


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import copy
import os

class NetworkMeta(nn.Module):
    def __init__(self, model_setup, env_tag, successSimulation, data_path, logname, lr, mlr, mo_lr, lw_atn, lw_w, lw_trj, lw_gen_trj, lw_dt, lw_phs, lw_fod, log_freq=25, gamma_sl = 0.995, device = 'cuda', use_transformer = True, tboard=True):
        super().__init__()
        self.optimizer         = None
        self.total_steps       = 0
        self.logname           = logname
        self.lr = lr
        self.mlr = mlr
        self.mo_lr = mo_lr
        self.device = device
        self.data_path = data_path
        self.use_transformer = use_transformer
        self.use_tboard = tboard
        self.embedding_memory = {}
        self.env_tag = env_tag
        self.init_train = True
        self.max_success_rate = 0
        self.max_step_disc = 12000
        self.num_vals = 0

        if self.logname.startswith("Intel$"):
            self.instance_name = self.logname.split("$")[1]
            self.logname       = self.logname.split("$")[0]
        else:
            self.instance_name = None

        if tboard:
            self.tboard            = TBoardGraphsTorch(self.logname, data_path=data_path)
        else:
            self.tboard = None
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

        self.successSimulation = successSimulation
        self.trajectories = None
        self.inpt_obs= None
        self.success= None

        self.last_mean_success = 0

        self.best_improved_success = 0

        self.best_mean_success = 0
        self.best_mean_success_complete = 0
        self.success_val = None
        self.model_setup = model_setup
        self.model_conv = ConvergenceDetector(setup=self.setup_model, set_meta=self.set_meta_module, writer=self.write_tboard_scalar, name='model', conf=0.05, interval = 300)
        self.tailor_conv = ConvergenceDetector(setup=self.setup_tailor, set_meta=self.set_meta_module, writer=self.write_tboard_scalar, name='tailor', conf = 0.1, interval = 100)
        self.max_acc = 0
        self.min_loss_module_train = float('inf')

    def setup_model(self):
        self.model   = PolicyTranslationModelTorch(od_path="", model_setup=self.model_setup, device=self.device)
        with torch.no_grad():
            for step, (d_in, d_out) in enumerate(self.train_loader):
                result = self.model(inputs=d_in)
                break
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, self.gamma_sl, verbose=False)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, 0.9, verbose=True)
        self.model_state_dict = self.model.state_dict()
        
    def set_meta_module(self):
        self.signal_main = SignalModule(model=self.model, loss_fct=self.calculateLoss)
        self.meta_module = MetaModule(main_signal = self.signal_main, tailor_signals=self.tailor_modules, lr = self.mo_lr, writer = self.write_tboard_scalar, device='cuda')

    def setup_tailor(self, set_tailor_data=False):
        policy = self.model
        gt_policy_success = False
        while not gt_policy_success:
            envs = self.successSimulation.get_env(n=2, env_tag = self.env_tag)
            result = self.successSimulation.get_success(policy = policy, envs=envs)
            if result is not False:
                trajectories, inpt_obs, label, success, ftrj = result
                gt_policy_success = True

        inpt_obs = inpt_obs.repeat((1, trajectories.size(1), 1))
        inpt = torch.concat((trajectories, inpt_obs), dim = -1)
        self.tailor_setup_inpt = inpt
        tailor_model = TailorTransformer(model_setup=self.model_setup['tailor_transformer'])
        tailor_models = [tailor_model]
        self.tailor_models = torch.nn.ModuleList(tailor_models)
        with torch.no_grad():
            for tmodel in self.tailor_models:
                t_result = tmodel.forward(inpt)        

        def lfp(result, label):
            return ((result.reshape(-1)-label.reshape(-1))**2).mean()
        self.tailor_modules = []
        for i in range(len(self.tailor_models)):
            self.tailor_modules.append(TaylorSignalModule(model=self.tailor_models[i], loss_fct=lfp, lr = self.mlr, mlr = self.mlr))
        if set_tailor_data:
            self.set_meta_module()
            self.setTailorDataset()
        
        
    def setTailorDataset(self):
        policy = self.meta_module
        policy.eval()
        policy.return_mode = 1
        gt_policy_success = False
        while not gt_policy_success:
            envs = self.successSimulation.get_env(n=2, env_tag = self.env_tag)
            result = self.successSimulation.get_success(policy = policy, envs=envs)
            if result is not False:
                trajectories, inpt_obs, label, success, achived_success = result
                gt_policy_success = True
        self.trajectories = trajectories
        self.inpt_obs = inpt_obs
        self.success = success
        self.tailor_data_train = None
        self.tailor_data_val = None
        for step, (d_in, d_out) in enumerate(self.train_loader):
            self.inpt_obs = torch.cat((self.inpt_obs, d_in[:,:1]), dim = 0)
            self.trajectories = torch.cat((self.trajectories, d_out))
            self.success = torch.cat((self.success, torch.ones(d_in.size(0), device = d_in.device)))
        data = (self.trajectories[:2], self.inpt_obs[:2], self.success[:2])
        self.add_data_to_tailor(data, split=0.5)
        data = (self.trajectories[2:], self.inpt_obs[2:], self.success[2:])
        self.add_data_to_tailor(data)

    def load_network(self, path_to_file, device = 'cuda'):
        pnp = path_to_file + "policy_network"
        tnp = path_to_file + "tailor_network"
        tdsp = path_to_file + "train_ds"
        tailor_train_p = path_to_file + "tailor_train"
        tailor_val_p = path_to_file + "tailor_val"

        self.train_ds = torch.load(tdsp, map_location=device)
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=16, shuffle=True)
        print(f'loaded train ds: {len(self.train_ds)}')
        self.model.load_state_dict(torch.load(pnp, map_location=device))
        self.tailor_modules[0].model.load_state_dict(torch.load(tnp, map_location=device))
        self.tailor_data_train = torch.load(tailor_train_p, map_location=device)
        self.tailor_data_val = torch.load(tailor_val_p, map_location=device)
        self.tailor_loader_train = torch.utils.data.DataLoader(self.tailor_data_train, batch_size=16, shuffle=True)
        self.tailor_loader_val = torch.utils.data.DataLoader(self.tailor_data_val, batch_size=16, shuffle=True)
        print(f'loaded train tailor: {len(self.tailor_data_train.success)}')
        self.init_train = False


    def saveNetworkToFile(self, add, data_path):
        print(f'save: {add}')
        print(f'{len(self.tailor_data_train)}')
        import pickle
        import os
        #dir_path = path.dirname(path.realpath(__file__))
        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not path.exists(path_to_file):
            makedirs(path_to_file)
        torch.save(self.model.state_dict(), path_to_file + "policy_network")
        with open(path_to_file + 'model_setup.pkl', 'wb') as f:
            pickle.dump(self.model.model_setup, f)  
        torch.save(self.train_ds, path_to_file + "train_ds")
        torch.save(self.tailor_data_train, path_to_file + "tailor_train")
        torch.save(self.tailor_data_val, path_to_file + "tailor_val")
        torch.save(self.tailor_modules[0].model.state_dict(), path_to_file + 'tailor_network')        

    def loadTailorDataset(self, path):
        obs_path = path + 'obs'
        trj_path = path + 'trj'
        s_path = path + 'success'
        if os.path.exists(s_path):
            self.inpt_obs = torch.load(obs_path)
            self.trajectories = torch.load(trj_path)
            self.success = torch.load(s_path)

    def setDatasets(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.train_ds = train_loader.dataset


    def train_tailor(self, ):
        trajectories, inpt_obs, success, ftrjs = self.successSimulation(policy = self.model, env_tag = self.env_tag, n = 10)
        inpt = torch.concat((trajectories, inpt_obs), dim = 0)
        debug_dict = tailor_optimizer(tailor_modules=self.tailor_modules, inpt=inpt, label=success)
        self.write_tboard_train_scalar(debug_dict=debug_dict)


    def train(self, epochs, model_params):
        num_convs_tailor = 0
        self.expert_examples_len = len(self.train_ds)
        print(f'inital num examples: {self.expert_examples_len}')
        self.global_step = 0
        #self.runValidation(quick=False, model_params=model_params)
        self.best_loss = float('inf')
        self.best_model = copy.deepcopy(self.model.state_dict())
        self.do_train_tailor = True
        self.best_tailor_model = copy.deepcopy(self.tailor_modules[0].model.state_dict())
        for epoch in range(epochs):
            self.model.reset_memory()
            print("Epoch: {:3d}/{:3d}".format(epoch+1, epochs)) 
            
            
            self.model.model_setup['train'] = True

            if not self.init_train:
                i = 0
                first_epoch = True
                while ((i < 10) and (self.min_loss_module_train > 1e-3)) or first_epoch:
                    first_epoch = False
                    i+=1
                    loss_module_train, debug_dict_train = self.tailor_epoch(loader=self.tailor_loader_train, train=True)
                    if loss_module_train < self.min_loss_module_train:
                        self.min_loss_module_train = loss_module_train
                    loss_module_val, debug_dict_val = self.tailor_epoch(loader=self.tailor_loader_val, train=False)
                    num_convs_tailor, conv_detected = self.tailor_conv.add_loss(torch.tensor(loss_module_train).reshape(1))
                    if debug_dict_val['acc95'] >= self.max_acc:
                            self.max_acc = debug_dict_val['acc95']
                            self.best_tailor_model = copy.deepcopy(self.tailor_modules[0].model.state_dict())
                            acc_dict = {"best acc": torch.tensor(self.max_acc)}
                            self.write_tboard_scalar(debug_dict=acc_dict, train=False)
                    if conv_detected:
                        if num_convs_tailor == 1:
                            print('tailor reset _________________________________________________')
                        break

            self.model.train()
            
            i = 0
            mean_train_loss = 10
            num_convs_model = 0
            first_epoch = True
            while ((i < 10) and (mean_train_loss > 1e-3) and (num_convs_model < 2)) or first_epoch:
                first_epoch = False
                num_convs_model, mean_train_loss = self.model_epoch(loader=self.train_loader)
                i += 1

            loss_goal_achieved = (self.best_loss < 1e-3) and ((self.min_loss_module_train < 1e-3) or self.init_train)
            val_epoch = (epoch+1) % model_params['val_every'] == 0
            max_epochs = (epoch+1) % (10*model_params['val_every']) == 0

            if (val_epoch and loss_goal_achieved) or (num_convs_model > 1) or (num_convs_tailor > 1) or (max_epochs and not self.init_train):
                self.num_vals += 1
                self.model.load_state_dict(self.best_model)
                self.tailor_modules[0].model.load_state_dict(self.best_tailor_model)
                self.best_loss = float('inf')
                self.min_loss_module_train = float('inf')
                self.min_loss_module_val = float('inf')
                complete = (self.num_vals == 5) or self.init_train
                if complete:
                    self.num_vals = 0
                print(f'logname: {self.logname}')
                #self.load_state_dict(self.best_model)
                self.runValidation(quick=False, epoch=epoch, save=True, model_params=model_params, complete=complete)
                self.init_train = False
           #self.train_tailor()
            
            self.scheduler.step()

    def model_epoch(self, loader):
        train_loss = []
        for step, (d_in, d_out) in enumerate(loader):
            self.global_step += 1
            loss = self.step(d_in, d_out, train=True)
            train_loss.append(loss)
            #self.loadingBar(self.total_steps, self.total_steps, 25, addition="Loss: {:.6f}".format(np.mean(train_loss)), end=True)
            num_convs_model, conv_detected_model = self.model_conv.add_loss(loss.reshape(1))
            if conv_detected_model and num_convs_model == 1:
                print('reset model ______________________________________________________')
                self.write_tboard_scalar({'num convs': torch.tensor(num_convs_model)}, train=False)
        mean_train_loss = torch.tensor(train_loss).mean()
        if mean_train_loss < self.best_loss:
            self.best_loss = mean_train_loss
            print(f'best model with {mean_train_loss}')
            self.best_model = copy.deepcopy(self.model.state_dict())
        return num_convs_model, mean_train_loss

    def tailor_epoch(self, loader, train):
        self.meta_module.train()
        self.model.eval()
        tloss = None
        acc = None
        for batch in loader:
            self.global_step += 1
            debug_dict = self.tailor_step(batch, train)
            if tloss is None:
                tloss = debug_dict['tailor loss'].reshape(1)
                acc = torch.tensor(debug_dict['tailor acc 0.950'].reshape(1))
            else:
                tloss = torch.cat((tloss, debug_dict['tailor loss'].reshape(1)), 0)
                acc = torch.cat((acc, torch.tensor(debug_dict['tailor acc 0.950'].reshape(1))), 0)

        #debug_dict = self.runvalidationTaylor()
        loss_module = tloss.mean()
        debug_dict['tailor module loss'] = loss_module
        debug_dict['acc95'] = acc.type(torch.float).mean()
        self.write_tboard_scalar(debug_dict=debug_dict, train=train)
        return loss_module, debug_dict
    
    def tailor_step(self, batch, train):
        debug_dict = tailor_optimizer(tailor_modules = self.tailor_modules, batch=batch, train=train)
        self.write_tboard_scalar(debug_dict=debug_dict, train=train)
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

    def runvalidationTaylor(self, debug_dict={}, return_mode = 0, num_exp = 2):
        self.meta_module.eval()
        self.model.eval()
        policy = self.meta_module
        if return_mode < 2:
            policy.return_mode = return_mode
        else:
            policy.return_mode = 0
        gt_policy_success = False
        while not gt_policy_success:
            envs = self.successSimulation.get_env(n=num_exp, env_tag = self.env_tag)
            result = self.successSimulation.get_success(policy = policy, envs=envs, her=False)
            if result is not False:
                trajectories, inpt_obs, label, success, ftrj = result
                gt_policy_success = True

        if return_mode == 2:
            trajectories = label
            success = torch.ones_like(success)
        fail = ~success
        taylor_inpt = {'result':trajectories, 'inpt':inpt_obs, 'original':ftrj}
        for i, ts in enumerate(policy.tailor_signals):
            expected_success = ts.forward(taylor_inpt)
            #expected_success = self.tailor_modules[0].forward(taylor_inpt)

            expected_success = expected_success.max(dim=-1)[1].reshape(-1).type(torch.bool)
            expected_fail = ~ expected_success
            expected_success = expected_success.type(torch.float)
            expected_fail = expected_fail.type(torch.float)

            fail = fail.type(torch.float).reshape(-1)
            success = success.type(torch.float).reshape(-1)
            tp = (expected_success * success)[success==1].mean()
            if success.sum() == 0:
                tp = torch.tensor(0)
            fp = (expected_success * fail)[fail==1].mean()
            tn = (expected_fail * fail)[fail==1].mean()
            fn = (expected_fail * success)[success==1].mean()

            if return_mode == 0:
                add = ' '+str(i)
            elif return_mode == 1:
                add = ' optimized '+str(i)
            elif return_mode == 2:
                add = ' label '+str(i)

            debug_dict['true positive' + add] = tp
            debug_dict['false positive' + add] = fp
            debug_dict['true negative' + add] = tn
            debug_dict['false negative' + add] = fn
            debug_dict['tailor success' + add] = (expected_success==success).type(torch.float).mean()
            debug_dict['tailor expected success' + add] = (expected_success).type(torch.float).mean()

        return debug_dict

    def add_data_to_tailor(self, data, split=0.8):
        trajectories, inpt_obs, success = data
        trj_len = int(split*len(trajectories))
        if self.tailor_data_train is not None:
            self.tailor_data_train.add_data(trajectories[:trj_len], inpt_obs[:trj_len], success[:trj_len])
            self.tailor_data_val.add_data(trajectories[trj_len:], inpt_obs[trj_len:], success[trj_len:])
        else:
            self.tailor_data_train = TorchDatasetTailor(trajectories[:trj_len], inpt_obs[:trj_len], success[:trj_len])
            self.tailor_data_val = TorchDatasetTailor(trajectories[trj_len:], inpt_obs[trj_len:], success[trj_len:])
        self.tailor_loader_train = DataLoader(self.tailor_data_train, batch_size=20, shuffle=True)
        self.tailor_loader_val = DataLoader(self.tailor_data_val, batch_size=20, shuffle=True)

    def runValidation(self, quick=False, pnt=False, epoch = 0, save = False, model_params = {}, complete = False): 
        self.meta_module.eval()
        self.model.eval()
        model_params = copy.deepcopy(model_params) #dont change model params globally
        self.model.model_setup['train'] = False
        #with torch.no_grad():
        if (not quick):
            if complete:
                num_envs = 100
            else:
                num_envs = 10
            #torch.manual_seed(1)
            print("Running full validation...")
            num_examples = int(self.tailor_data_train.num_ele())
            if complete and not self.init_train:
                print('complete:')
                self.meta_module.optim_run += 1
                debug_dict = self.runvalidationTaylor(num_exp=num_envs)
                self.write_tboard_scalar(debug_dict=debug_dict, train = False, step=num_examples)
                debug_dict = self.runvalidationTaylor(return_mode=1, num_exp=num_envs)
                tailor_success_optimized = debug_dict['tailor success optimized 0']

                #tailor_success_optimized = debug_dict['true positive optimized']
                self.write_tboard_scalar(debug_dict=debug_dict, train = False, step=num_examples)
                debug_dict = self.runvalidationTaylor(return_mode=2, num_exp=num_envs)
                self.write_tboard_scalar(debug_dict=debug_dict, train = False, step=num_examples)

            policy = self.meta_module
            policy.main_signal.model.eval()
            policy.return_mode = 0
            gt_policy_success = False
            while not gt_policy_success:
                envs = self.successSimulation.get_env(n=num_envs, env_tag = self.env_tag)
                result = self.successSimulation.get_success(policy = policy, envs=envs)
                if result is not False:
                    trajectories, inpt_obs, label, success, achieved_succ = result
                    gt_policy_success = True

            print(f'num envs: {len(envs)}')
            fail = ~success
            #mean_success = success[:int(len(success)/2)].type(torch.float).mean()
            mean_success = achieved_succ.type(torch.float).mean()

            print(f'mean success before: {mean_success}')
            debug_dict = {'success rate generated' : mean_success}
            policy.return_mode = 1
            
            if not self.init_train:
                trajectories_opt, inpt_obs_opt, label_opt, success_opt, achieved_succ_opt = self.successSimulation.get_success(policy = policy, envs=envs)
                fail_opt = ~success_opt
                if len(success_opt)>0:
                    #mean_success_opt = success_opt[:int(len(success)/2)].type(torch.float).mean()
                    mean_success_opt = achieved_succ_opt.type(torch.float).mean()
                else:
                    mean_success_opt = 0
                if mean_success_opt > self.last_mean_success:
                    policy.max_steps = policy.max_steps * 1.1
                    self.last_mean_success = mean_success_opt


                
                '''if tailor_success_optimized * 0.8 > mean_success_opt:
                    policy.max_steps = policy.max_steps * 1.1
                elif tailor_success_optimized < mean_success_opt:
                    policy.max_steps = max(policy.max_steps * 0.9, 5)'''

                self.write_tboard_scalar({'num optimisation steps':torch.tensor(policy.max_steps)}, train= False)

                print(f'mean success after: {mean_success_opt}')
                debug_dict['success rate optimized'] = mean_success_opt
                debug_dict['improved success rate'] = mean_success_opt - mean_success
                if mean_success_opt - mean_success > self.best_improved_success:
                    self.best_improved_success = mean_success_opt - mean_success
                    if complete:
                        self.saveNetworkToFile(add=self.logname + "/best_improved_complete/", data_path= self.data_path)
                    else:
                        self.saveNetworkToFile(add=self.logname + "/best_improved/", data_path= self.data_path)

                '''num_improved = (success_opt * fail).type(torch.float).mean()
                num_deproved = (success * fail_opt).type(torch.float).mean()
                debug_dict['rel number improved'] = num_improved
                debug_dict['rel number failed'] = num_deproved'''

            self.write_tboard_scalar(debug_dict=debug_dict, train=not complete, step = self.global_step)
    
            if not complete:
                self.add_data_to_tailor((trajectories, inpt_obs, success))
                self.add_data_to_tailor((trajectories_opt, inpt_obs_opt, success_opt))

                
                print(f'num tailor example: {len(self.tailor_data_train.success)}')
                print(f'num tailor example val: {len(self.tailor_data_val.success)}')

                if not self.init_train:
                    #train_data.add_data(data=inpt_obs[success], label=trajectories[success])
                    self.train_ds.add_data(data=inpt_obs_opt[success_opt], label=trajectories_opt[success_opt])
                    print('added train data')
                
                self.model_conv.reset_history()
                self.tailor_conv.reset_history()
                self.max_acc = 0
                '''self.model.plan_nn = None
                self.setup_model(set_tailor_data=False)
                print('reset model')'''
                self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=16, shuffle=True)

                #print(f'num examples: {len(self.success)}')
                print(f'num demonstrations: {len(self.train_ds)}')
                self.write_tboard_scalar({'num examples':torch.tensor(self.tailor_data_train.num_ele())}, train=True)
                self.write_tboard_scalar({'num examples val':torch.tensor(self.tailor_data_val.num_ele())}, train=True)
                self.model_conv.num_convs = 0
                self.tailor_conv.num_convs = 0
            else:
                if mean_success >= self.best_mean_success_complete:
                    self.best_mean_success_complete = mean_success
                    self.best_model_val = copy.deepcopy(self.model.state_dict())
                    print(f'saved model with: {mean_success}')
                else:
                    self.model.load_state_dict(self.best_model_val)
                    print('reloaded model')
        if mean_success >= self.best_mean_success:
            self.best_mean_success = mean_success
            if complete:
                self.saveNetworkToFile(add=self.logname + "/best_generated_complete/", data_path= self.data_path)
            else:
                self.saveNetworkToFile(add=self.logname + "/best_generated/", data_path= self.data_path)

        val_loss = []
        for step, (d_in, d_out) in enumerate(self.val_loader):
            loss = self.step(d_in, d_out, train=False)
            val_loss.append(loss)
            if quick or model_params['quick_val']:
                break

        val_loss = torch.tensor(val_loss).mean()

        if self.use_tboard:
            do_dim = d_in[0].size(0)
            #print(d_in)
            #self.model.eval()
            
            if not quick:
                if 'meta_world' in self.model.model_setup and self.model.model_setup['meta_world']['use']:

                    self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=success, name = 'success')

                    fail = ~success
                    self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=fail, name = 'fail')

                    if not self.init_train:
                        fail_opt = ~success_opt

                        fail_to_success = success_opt & fail
                        self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=fail_to_success, name = 'fail to success', opt_trj=trajectories_opt)

                        success_to_fail = success & fail_opt
                        self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=success_to_fail, name = 'success to fail', opt_trj=trajectories_opt)
                        
                        fail_to_fail = fail & fail_opt
                        self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=fail_to_fail, name = 'fail to fail', opt_trj=trajectories_opt)

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

        if pnt:
            print("  Validation Loss: {:.6f}".format(val_loss))
        if not quick:
            if loss < self.global_best_loss_val:
                self.global_best_loss_val = loss
        return val_loss

    def write_tboard_scalar(self, debug_dict, train, step = None):
        if step is None:
            step = self.global_step
        if self.use_tboard:
                for para, value in debug_dict.items():
                    if train:
                        self.tboard.addTrainScalar(para, value, step)
                    else:
                        self.tboard.addValidationScalar(para, value, step)

    def plot_with_mask(self, label, trj, inpt, mask, name, opt_trj=None):
        if mask.sum()>0:
            label = label[mask][0]
            trj = trj[mask][0]
            inpt = inpt[mask][0,0]
            if opt_trj is not None:
                opt_trj = opt_trj[mask][0]
            self.createGraphsMW(d_in=1, d_out=label, result=trj, toy=False, inpt=inpt, name=name, opt_trj=opt_trj, window=self.successSimulation.window)

    def step(self, d_in, d_out, train):

        if train:
            result = self.model(d_in)
            loss, debug_dict = self.calculateLoss(d_out, result)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.write_tboard_scalar(debug_dict=debug_dict, train=True)
        else:
            with torch.no_grad():
                result = self.model(d_in)
                loss, debug_dict = self.calculateLoss(d_out=d_out, result=result)
            

            if self.last_written_step != self.global_step:
                if self.use_tboard:
                    self.last_written_step = self.global_step
                    self.write_tboard_scalar(debug_dict=debug_dict, train=False)

                loss = loss.detach().cpu()
                if loss < self.global_best_loss:
                    self.global_best_loss = loss
                    if self.use_tboard:
                        self.saveNetworkToFile(add=self.logname + "/best/", data_path= self.data_path)
                
                    #print(f'model saved with loss: {loss}')

        return loss.detach()
    
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

    def calculateLoss(self, d_out, result, prefix = ''):
        if 'meta_world' in self.model.model_setup and self.model.model_setup['meta_world']['use']:
            generated = d_out
        else:
            generated, attention, delta_t, weights, phase, loss_atn = d_out
            atn = result['atn']
            atn_loss = self.catCrossEntrLoss(attention, atn)
            rel_correct_objects = (torch.argmax(atn, dim=-1) == torch.argmax(attention, dim=-1)).sum()/len(atn)


        gen_trj = result['gen_trj']
        #phs = result['phs']

        #phs_loss = self.calcMSE(phase, phs)
        trj_loss = self.calcMSE(generated, gen_trj)

        debug_dict = {
            prefix + 'trj_loss':trj_loss,
            #prefix + 'phs_loss':phs_loss,
            }
        if 'meta_world' in self.model.model_setup and self.model.model_setup['meta_world']['use']:
            loss = trj_loss * self.lw_trj
            debug_dict[prefix + 'main_loss'] = loss
        
        else:
            loss = atn_loss * self.lw_atn + \
                    trj_loss * self.lw_trj
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
        
    def createGraphsMW(self, d_in, d_out, result, save = False, name_plot = '', epoch = 0, model_params={}, toy=True, inpt=None, name='Trajectory', opt_trj = None, window = 0):
        target_trj  = d_out
        gen_trj = result
        
        path_to_plots = self.data_path + "/plots/"+ str(self.logname) + '/' + str(epoch) + '/'
        if toy:
            tol_neg = self.successSimulation.neg_tol
            tol_pos = self.successSimulation.pos_tol
        else:
            tol_neg = None
            tol_pos = None
        #gen_tr_trj= self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_trj), axis=0))
        #gen_tr_phase = self.tf2torch(tf.math.reduce_mean(self.torch2tf(phase), axis=0))
        self.tboard.plotDMPTrajectory(target_trj, gen_trj, torch.zeros_like(gen_trj),
                                    None, None, None, stepid=self.global_step, save=save, name_plot=name_plot, path=path_to_plots,\
                                        tol_neg=tol_neg, tol_pos=tol_pos, inpt = inpt, name=name, opt_gen_trj = opt_trj, window=window)
        



