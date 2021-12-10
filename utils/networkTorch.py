# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals
from os import path
from pickle import NONE
import tensorflow as tf
import sys
import numpy as np
from torch._C import dtype
from utils.graphsTorch import TBoardGraphsTorch
import tensorflow as tf

import torch
import torch.nn as nn
import time

class NetworkTorch(nn.Module):
    def __init__(self, model, data_path, logname, lr, lw_atn, lw_w, lw_trj, lw_gen_trj, lw_dt, lw_phs, lw_fod, log_freq=25, gamma_sl = 0.995, device = 'cuda', use_transformer = True, tboard=True):
        super().__init__()
        self.optimizer         = None
        self.model             = model
        self.total_steps       = 0
        self.logname           = logname
        self.lr = lr
        self.device = device
        self.data_path = data_path
        self.use_transformer = use_transformer
        self.use_tboard = tboard

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

    def setup_model(self):
        with torch.no_grad():
            for step, (d_in, d_out) in enumerate(self.train_ds):
                result = self.model(inputs=d_in, training=True)
                break

    def setDatasets(self, train_loader, val_loader):
        self.train_ds = train_loader
        self.val_ds   = val_loader

    def train(self, epochs, use_transformer):
        self.global_step = 0
        for epoch in range(epochs):
            rel_epoch = epoch/epochs
            print("Epoch: {:3d}/{:3d}".format(epoch+1, epochs)) 
            validation_loss = 0.0
            train_loss = []
            teb = rel_epoch < 0.3
            print(f'train ambedding: {teb}')
            for step, (d_in, d_out) in enumerate(self.train_ds):
                if step % 100 == 0:
                    validation_loss = self.runValidation(quick=True, pnt=False)                    
                train_loss.append(self.step(d_in, d_out, train=True, train_embedding=teb))
                
                self.loadingBar(step, self.total_steps, 25, addition="Loss: {:.6f} | {:.6f}".format(np.mean(train_loss[-10:]), validation_loss))
                if epoch == 0:
                    self.total_steps += 1
                self.global_step += 1
            self.loadingBar(self.total_steps, self.total_steps, 25, addition="Loss: {:.6f}".format(np.mean(train_loss)), end=True)

            self.runValidation(quick=False)
            self.scheduler.step()
            print(f'learning rate: {self.scheduler.get_last_lr()[0]}')

            if self.use_tboard:
                self.model.saveModelToFile(add = self.logname + "/", data_path = self.data_path)

            if epoch % self.log_freq == 0 and self.instance_name is not None:
                self._uploadToCloud(epoch)

        if self.instance_name is not None:
            self._uploadToCloud()
    
    
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
    def runValidation(self, quick=False, pnt=True): 
        if not quick:
            print("Running full validation...")
        val_loss = []
        for step, (d_in, d_out) in enumerate(self.val_ds):
            val_loss.append(self.step(d_in, d_out, train=False, recursive = True))
            if quick:
                break
        in0 = self.tf2torch(tf.tile(tf.expand_dims(self.torch2tf(d_in[0][0]), 0),[50,1]))
        in1 = self.tf2torch(tf.tile(tf.expand_dims(self.torch2tf(d_in[1][0]), 0),[50,1,1]))
        in2 = self.tf2torch(tf.tile(tf.expand_dims(self.torch2tf(d_in[2][0]), 0),[50,1,1]))
        d_in_graphs  = (in0, in1, in2)
        #self.model.eval()
        out_model = self.model(d_in_graphs, training=True, return_gen_gen=True)
        #self.model.eval()
        #out_model_tf = [self.torch2tf(out_model[0]), [self.torch2tf(ele) for ele in out_model[1]]]
        #d_out_graphs = (tf.tile(tf.expand_dims(d_out[0][0], 0),[50,1,1]), tf.tile(tf.expand_dims(d_out[1][0], 0),[50,1]), 
        #                tf.tile(tf.expand_dims([d_out[2][0]], 0),[50,1]), tf.tile(tf.expand_dims(d_out[3][0], 0),[50,1,1]))
        if self.use_tboard:
            self.createGraphs((d_in[0][0], d_in[1][0], d_in[2][0]),
                            (d_out[0][0], d_out[1][0], d_out[2][0], d_out[3][0]), 
                            out_model,
                            save=True)
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

    def step(self, d_in, d_out, train, train_embedding = True, recursive = False):
        generated, attention, delta_t, weights, phase, loss_atn = d_out
        if not train:
            self.model.eval()
        result = self.model(d_in, training=train, gt_attention = attention, train_embedding=train_embedding, return_gen_gen = True, recursive=recursive)
        if not train:
            self.model.train()
        loss, (atn, trj, dt, phs, wght, rel_obj) = self.calculateLoss(d_out, result, train)


        if train:
            if not self.optimizer:
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2) 
                #self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999)) 
                #self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, weight_decay=1e-2) 
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=self.gamma_sl)
            #print(f'num parametrs in model: {len(list(self.model.parameters()))}')

            self.optimizer.zero_grad()
            h = time.perf_counter()

            loss.backward()

            '''for i, para in enumerate(list(self.model.parameters())):
                print(f'para num {i} has grad sum {para.grad.detach().sum()}')'''
            self.optimizer.step()

            if self.use_tboard:
                self.tboard.addTrainScalar("Loss", loss, self.global_step)
                self.tboard.addTrainScalar("Loss Attention", atn, self.global_step)
                self.tboard.addTrainScalar("Loss Trajectory", trj, self.global_step)
                self.tboard.addTrainScalar("Loss Phase", phs, self.global_step)
                self.tboard.addTrainScalar("Loss Weight", wght, self.global_step)
                self.tboard.addTrainScalar("Loss Delta T", dt, self.global_step)
                self.tboard.addTrainScalar("Relative Object correct", rel_obj, self.global_step)
        else:
            if self.last_written_step != self.global_step:
                if self.use_tboard:
                    self.last_written_step = self.global_step
                    self.tboard.addValidationScalar("Loss", loss, self.global_step)
                    self.tboard.addValidationScalar("Loss Attention", atn, self.global_step)
                    self.tboard.addValidationScalar("Loss Trajectory", trj, self.global_step)
                    self.tboard.addValidationScalar("Loss Phase", phs, self.global_step)
                    self.tboard.addValidationScalar("Loss Weight", wght, self.global_step)
                    self.tboard.addValidationScalar("Loss Delta T", dt, self.global_step)
                    self.tboard.addValidationScalar("Relative Object correct", rel_obj, self.global_step)

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

    def calculateLoss(self, d_out, result, train):
        generated, attention, delta_t, weights, phase, loss_atn = d_out
        if self.use_transformer:
            gen_trj, gen_gen_trj, atn, phs                          = result

        else:
            gen_trj, (atn, dmp_dt, phs, wght)                       = result  #gen trj = 32, 350, 7
        #fod_loss = gen_trj
        #weight_dim  = torch.tensor([3.0, 3.0, 3.0, 1.0, 0.5, 1.0, 0.1], device = generated.device)
        weight_dim  = torch.tensor([1.0, 1.0, 1.0, 1.0, 1, 1.0, 1.], device = generated.device)

        atn_loss = self.catCrossEntrLoss(attention, atn)

        fod_loss = nn.MSELoss()(gen_trj[:,1:], gen_trj[:, :-1])
        rel_correct_objects = (torch.argmax(atn, dim=-1) == torch.argmax(attention, dim=-1)).sum()/len(atn)
        if not self.use_transformer:
            dt_loss = self.mse_loss(delta_t, dmp_dt[:,0]).mean()
            #TODO why :-1?'''
            weight_loss = nn.MSELoss(reduction='none')(wght[:,:-1,:,:], wght[:,:-1,:,:].roll(shifts = -1, dims = 1)).mean((-2,-1))
            weight_loss = (weight_loss * loss_atn[:,:-1]).mean()

        repeated_weight_dim = weight_dim.reshape(1,1,-1).repeat([gen_trj.size(0), gen_trj.size(1), 1])
        phs_loss = self.calculateMSEWithPaddingMask(phase, phs.squeeze(), loss_atn).mean()
        #trj_loss = ((generated- gen_trj)**2).mean()
        #trj_loss = trj_loss.mean()
        trj_loss = self.calculateMSEWithPaddingMask(generated, gen_trj, repeated_weight_dim)
        trj_gen_loss = self.calculateMSEWithPaddingMask(generated, gen_gen_trj, repeated_weight_dim)
        trj_loss = (trj_loss * loss_atn).mean()
        trj_gen_loss = (trj_gen_loss * loss_atn).mean()

        if self.use_transformer:
            return (atn_loss * self.lw_atn + 
                    trj_loss * self.lw_trj + 
                    trj_gen_loss * self.lw_gen_trj +
                    phs_loss * self.lw_phs + 
                    self.lw_fod * fod_loss, 
                    (atn_loss, trj_loss, fod_loss, phs_loss, trj_gen_loss, rel_correct_objects))
        else:
            return (atn_loss * self.lw_atn +
                    trj_loss * self.lw_trj +
                    phs_loss * self.lw_phs +
                    weight_loss * self.lw_w + 
                    dt_loss  * self.lw_dt,
                    (atn_loss, trj_loss, dt_loss, phs_loss, weight_loss, rel_correct_objects)
                )
    
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

    def createGraphs(self, d_in, d_out, result, save = False):
        language, image, robot_states            = d_in
        target_trj, attention, delta_t, weights  = d_out
        if not self.use_transformer:
            gen_trj, (atn, dmp_dt, phase, wght)      = result
        else:
            gen_trj, gen_gen_trj, atn, phase = result

            dmp_dt = None



        self.tboard.plotClassAccuracy(attention, self.tf2torch(tf.math.reduce_mean(self.torch2tf(atn), axis=0)), self.tf2torch(tf.math.reduce_std(self.torch2tf(atn), axis=0)), self.tf2torch(self.torch2tf(language)), stepid=self.global_step)
        if not self.use_transformer:
            self.tboard.plotDMPTrajectory(target_trj, self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_trj), axis=0)), self.tf2torch(tf.math.reduce_std(self.torch2tf(gen_trj), axis=0)),
                                        self.tf2torch(tf.math.reduce_mean(self.torch2tf(phase), axis=0)), delta_t, self.tf2torch(tf.math.reduce_mean(self.torch2tf(dmp_dt), axis=0)), stepid=self.global_step)
        else:
            gen_tr_trj= self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_trj), axis=0))
            gen_gen_trj= self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_gen_trj), axis=0))
            gen_tr_phase = self.tf2torch(tf.math.reduce_mean(self.torch2tf(phase), axis=0))
            self.tboard.plotDMPTrajectory(target_trj, gen_tr_trj, torch.zeros_like(gen_tr_trj),
                                        gen_tr_phase, delta_t, None, stepid=self.global_step, save=save)
            self.tboard.plotDMPTrajectory(target_trj, gen_gen_trj, torch.zeros_like(gen_gen_trj),
                                        gen_tr_phase, delta_t, None, stepid=self.global_step, name='Gen - Trajectoy', save=save)

                

