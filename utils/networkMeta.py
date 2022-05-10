# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals
from os import name, path, makedirs
import tensorflow as tf
import sys
import numpy as np
from LanguagePolicies.utils.graphsTorch import TBoardGraphsTorch
from MetaWorld.utilsMW.metaOptimizer import SignalModule, TaylorSignalModule, meta_optimizer, tailor_optimizer, MetaModule
from MetaWorld.utilsMW.dataLoaderMW import TorchDatasetTailor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import copy

class NetworkMeta(nn.Module):
    def __init__(self, model, tailor_models, env_tag, successSimulation, data_path, logname, lr, mlr, mo_lr, lw_atn, lw_w, lw_trj, lw_gen_trj, lw_dt, lw_phs, lw_fod, log_freq=25, gamma_sl = 0.995, device = 'cuda', use_transformer = True, tboard=True):
        super().__init__()
        self.optimizer         = None
        self.model             = model
        self.tailor_models      = tailor_models
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

        self.successSimulation = successSimulation
        self.trajectories = None
        self.inpt_obs= None
        self.success= None



    def setup_model(self, model_params):
        with torch.no_grad():
            for step, (d_in, d_out) in enumerate(self.train_ds):
                result = self.model(inputs=d_in)
                break
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, self.gamma_sl, verbose=True)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, 0.9, verbose=True)

        self.signal_main = SignalModule(model=self.model, loss_fct=self.calculateLoss, optimizer=self.optimizer)
        policy = self.model
        envs = self.successSimulation.get_env(n=2)
        trajectories, inpt_obs, label, success = self.successSimulation.get_success(policy = policy, envs=envs)
        inpt_obs = inpt_obs.repeat((1, trajectories.size(1), 1))
        inpt = torch.concat((trajectories, inpt_obs), dim = -1)
        #inpt = inpt.transpose(0,1)
        with torch.no_grad():
            for tmodel in self.tailor_models:
                t_result = tmodel.forward(inpt)
        self.tailor_optimizers = [torch.optim.Adam(params=tailor_model.parameters(), lr=self.mlr) for tailor_model in self.tailor_models]
        self.meta_optimizers = [torch.optim.Adam(params=tailor_model.parameters(), lr=self.mlr) for tailor_model in self.tailor_models]
        #self.tailor_optimizers = [torch.optim.SGD(params=tailor_model.parameters(), lr=self.lr) for tailor_model in self.tailor_models]
        #self.meta_optimizers = [torch.optim.SGD(params=tailor_model.parameters(), lr=0.1*self.lr) for tailor_model in self.tailor_models]
        
        def lfp(result, label):
            return ((result.reshape(-1)-label.reshape(-1))**2).mean()
        self.tailor_modules = []
        for i in range(len(self.tailor_models)):
            self.tailor_modules.append(TaylorSignalModule(model=self.tailor_models[i], loss_fct=lfp, optimizer=self.tailor_optimizers[i], meta_optimizer=self.meta_optimizers[i]))
        self.model_state_dict = self.model.state_dict()
        self.meta_module = MetaModule(main_signal = self.signal_main, tailor_signal=self.tailor_modules[0], lr = self.mo_lr, writer = self.write_tboard_scalar)
        self.setTailorDataset()
        '''self.init_tailor_models = []
        for i in range(len(self.tailor_models)):
            self.init_tailor_models.append(copy.deepcopy(self.tailor_models[i]))

    def reset_tailor_models(self):
        for i in range(len(self.tailor_models)):
            self.tailor_models[i] = copy.deepcopy(self.init_tailor_models[i])'''

    def setTailorDataset(self):
        policy = self.meta_module
        policy.return_mode = 0
        envs = self.successSimulation.get_env(n=2)
        trajectories, inpt_obs, label, success = self.successSimulation.get_success(policy = policy, envs=envs)
        self.trajectories = trajectories
        self.inpt_obs = inpt_obs
        self.success = success
        for step, (d_in, d_out) in enumerate(self.train_ds):
            self.inpt_obs = torch.cat((self.inpt_obs, d_in[:,:1]), dim = 0)
            self.trajectories = torch.cat((self.trajectories, d_out))
            self.success = torch.cat((self.success, torch.ones(d_in.size(0), device = d_in.device)))

        self.inpt_obs = self.inpt_obs
        self.trajectories = self.trajectories
        self.success = self.success

        

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
                #self.reset_tailor_models()
                while loss_module > 0.2:
                    lmp = None
                    lmn = None
                    for succ, failed in self.tailor_loader:
                        self.global_step += 1
                        debug_dict = self.tailor_step(succ, failed)
                        if lmp is None:
                            lmp = debug_dict['tailor loss positive'].reshape(1)
                            lmn = debug_dict['tailor loss positive'].reshape(1)
                        else:
                            lmp = torch.cat((lmp, debug_dict['tailor loss positive'].reshape(1)), 0)
                            lmn = torch.cat((lmn, debug_dict['tailor loss negative'].reshape(1)), 0)

                        loss_module = torch.maximum(lmp, lmn).mean()
                    
                    debug_dict = self.runvalidationTaylor()
                    debug_dict['tailor module loss'] = loss_module
                    self.write_tboard_scalar(debug_dict=debug_dict, train=False)
                
                self.runValidation(quick=False, model_params=model_params)

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
                self.saveNetworkToFile(add = self.logname + "/", data_path = self.data_path)
            self.scheduler.step()

    
    
    def tailor_step(self, succ, failed):
        debug_dict = tailor_optimizer(tailor_modules = self.tailor_modules, succ=succ, failed=failed)
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

    def runvalidationTaylor(self, debug_dict={}, return_mode = 0):
        policy = self.meta_module
        if return_mode < 2:
            policy.return_mode = return_mode
        else:
            policy.return_mode = 0
        envs = self.successSimulation.get_env(n=300)
        trajectories, inpt_obs, label, success = self.successSimulation.get_success(policy = policy, envs=envs)    
        if return_mode == 2:
            trajectories = label
            success = torch.ones_like(success)
        fail = ~success
        taylor_inpt = {'result':trajectories, 'inpt':inpt_obs}

        with torch.no_grad():
            expected_success = self.tailor_modules[0].forward(taylor_inpt)

        expected_success = expected_success.max(dim=-1)[1].reshape(-1).type(torch.bool)
        expected_fail = ~ expected_success
        expected_success = expected_success.type(torch.float)
        expected_fail = expected_fail.type(torch.float)

        fail = fail.type(torch.float).reshape(-1)
        success = success.type(torch.float).reshape(-1)
        tp = (expected_success * success)[success==1].mean()
        fp = (expected_success * fail)[fail==1].mean()
        tn = (expected_fail * fail)[fail==1].mean()
        fn = (expected_fail * success)[success==1].mean()

        if return_mode == 0:
            add = ''
        elif return_mode == 1:
            add = ' optimized'
        elif return_mode == 2:
            add = ' label'

        debug_dict['true positive' + add] = tp
        debug_dict['false positive' + add] = fp
        debug_dict['true negative' + add] = tn
        debug_dict['false negative' + add] = fn
        debug_dict['tailor success' + add] = (expected_success==success).type(torch.float).mean()
        debug_dict['tailor expected success' + add] = (expected_success).type(torch.float).mean()

        return debug_dict

    def runValidation(self, quick=False, pnt=True, epoch = 0, save = False, model_params = {}): 
        model_params = copy.deepcopy(model_params) #dont change model params globally
        self.model.model_setup['train'] = False
        #with torch.no_grad():
        if (not quick):
            #torch.manual_seed(1)
            print("Running full validation...")
            self.meta_module.optim_run += 1
            debug_dict = self.runvalidationTaylor()
            self.write_tboard_scalar(debug_dict=debug_dict, train = False)
            debug_dict = self.runvalidationTaylor(return_mode=1)
            self.write_tboard_scalar(debug_dict=debug_dict, train = False)
            debug_dict = self.runvalidationTaylor(return_mode=2)

            policy = self.meta_module
            policy.return_mode = 0
            envs = self.successSimulation.get_env(n=400)
            trajectories, inpt_obs, label, success = self.successSimulation.get_success(policy = policy, envs=envs)

            mean_success = success.type(torch.float).mean()
            print(f'mean success before: {mean_success}')
            debug_dict = {'success rate generated' : mean_success}

            policy.return_mode = 1
            trajectories_opt, inpt_obs_opt, label_opt, success_opt = self.successSimulation.get_success(policy = policy, envs=envs)
            mean_success_opt = success_opt.type(torch.float).mean()
            print(f'mean success after: {mean_success_opt}')
            debug_dict['success rate optimized'] = mean_success_opt

            self.write_tboard_scalar(debug_dict=debug_dict, train=False)
        
            if mean_success > self.max_success_rate:
                self.max_success_rate = mean_success
            else:
                pass
                #self.model.load_state_dict(self.model_state_dict)
            num_exp = 20
            if mean_success > 1:
                self.trajectories = torch.cat((self.trajectories, trajectories[:num_exp]), dim=0)[-30000:]
                self.inpt_obs = torch.cat((self.inpt_obs, inpt_obs[:num_exp]), dim=0)[-30000:]
                self.success = torch.cat((self.success, success[:num_exp]), dim=0)[-30000:]

                '''self.trajectories = torch.cat((self.trajectories, trajectories_opt), dim=0)[-30000:]
                self.inpt_obs = torch.cat((self.inpt_obs, inpt_obs_opt), dim=0)[-30000:]
                self.success = torch.cat((self.success, success_opt), dim=0)[-30000:]'''
                print(f'num examples: {len(self.success)}')
                tailor_data = TorchDatasetTailor(trajectories= self.trajectories, obsv=self.inpt_obs, success=self.success)
                self.tailor_loader = DataLoader(tailor_data, batch_size=32, shuffle=True)
                self.init_train = False

            '''if mean_success > 0.1:
                self.init_train = False


            else:
                self.init_train = True
            '''



        val_loss = []
        for step, (d_in, d_out) in enumerate(self.val_ds):
            loss = self.step(d_in, d_out, train=False, model_params=model_params)
            val_loss.append(loss)
            if quick or model_params['quick_val']:
                break

        if self.use_tboard:
            do_dim = d_in[0].size(0)
            #print(d_in)
            #self.model.eval()
            
            if not quick:
                if 'meta_world' in self.model.model_setup and self.model.model_setup['meta_world']['use']:

                    self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=success, name = 'success')

                    fail = ~success
                    fail_opt = ~success_opt
                    self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=fail, name = 'fail')

                    fail_to_success = success_opt & fail
                    self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=fail_to_success, name = 'fail to success', opt_trj=trajectories_opt)

                    success_to_fail = success & fail_opt
                    self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=success_to_fail, name = 'success to fail', opt_trj=trajectories_opt)
                    
                    fail_to_fail = fail & fail_opt
                    self.plot_with_mask(label=label, trj=trajectories, inpt=inpt_obs, mask=fail_to_fail, name = 'fail to fail', opt_trj=trajectories_opt)

                    
                    #print(f'label-check: {label[success_to_fail][0]}')
                    #print(f'label-opt-check: {label_opt[success_to_fail][0]}')

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
                    self.saveNetworkToFile(add=self.logname + "/best_val/", data_path= self.data_path)
                    #self.model.saveModelToFile(add=self.logname + "/best_val/", data_path= self.data_path)
                    print(f'best val model saved with: {loss}')
        return np.mean(val_loss)

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



            self.createGraphsMW(d_in=1, d_out=label, result=trj, toy=True, inpt=inpt, name=name, opt_trj=opt_trj, window=self.successSimulation.window)

    def step(self, d_in, d_out, train, model_params):

        if train:
            if self.init_train or True:
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
                        self.saveNetworkToFile(add=self.logname + "/best/", data_path= self.data_path)
                
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
        

    def saveNetworkToFile(self, add, data_path):
        import pickle
        import os
        #dir_path = path.dirname(path.realpath(__file__))
        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not path.exists(path_to_file):
            makedirs(path_to_file)
        torch.save(self.state_dict(), path_to_file + "policy_network")
        with open(path_to_file + 'model_setup.pkl', 'wb') as f:
            pickle.dump(self.model.model_setup, f)  
        for name in self.model.memory:
            torch.save(self.model.memory[name], path_to_file + name)