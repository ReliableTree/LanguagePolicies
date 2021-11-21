# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import numpy as np
from utils.graphsTorch import TBoardGraphsTorch
from utils.network import Network
import tensorflow as tf

import torch
import torch.nn as nn
import time

class NetworkTorch(nn.Module):
    def __init__(self, model, logname, lr, lw_atn, lw_w, lw_trj, lw_dt, lw_phs, log_freq=25, gamma_sl = 0.995):
        super().__init__()
        self.tf_test_nw = Network(model, logname, lr, lw_atn, lw_w, lw_trj, lw_dt, lw_phs, log_freq=25)
        self.optimizer         = None
        self.model             = model
        self.total_steps       = 0
        self.logname           = logname
        self.lr = lr

        if self.logname.startswith("Intel$"):
            self.instance_name = self.logname.split("$")[1]
            self.logname       = self.logname.split("$")[0]
        else:
            self.instance_name = None

        self.tboard            = TBoardGraphsTorch(self.logname)
        self.loss              = nn.CrossEntropyLoss()
        self.global_best_loss  = float('inf')
        self.last_written_step = -1
        self.log_freq          = log_freq

        self.lw_atn = lw_atn 
        self.lw_w   = lw_w 
        self.lw_trj = lw_trj
        self.lw_dt  = lw_dt
        self.lw_phs = lw_phs

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.gamma_sl = gamma_sl

    def setup_model(self):
        with torch.no_grad():
            for step, (d_in, d_out) in enumerate(self.train_ds):
                result = self.model(d_in, training=True)
                break

    def setDatasets(self, train_loader, val_loader):
        self.train_ds = train_loader
        self.val_ds   = val_loader

    def train(self, epochs, result):
        self.global_step = 0
        for epoch in range(epochs):
            print("Epoch: {:3d}/{:3d}".format(epoch+1, epochs)) 
            validation_loss = 0.0
            train_loss = []
            for step, (d_in, d_out) in enumerate(self.train_ds):
                #if step % 100 == 0:
                #    validation_loss = self.runValidation(quick=True, pnt=False)    
                return self.step(d_in, d_out, train=True, result = result)            
                train_loss.append(self.step(d_in, d_out, train=True, result = result))
                self.loadingBar(step, self.total_steps, 25, addition="Loss: {:.6f} | {:.6f}".format(np.mean(train_loss[-10:]), validation_loss))
                if epoch == 0:
                    self.total_steps += 1
                self.global_step += 1
            self.loadingBar(self.total_steps, self.total_steps, 25, addition="Loss: {:.6f}".format(np.mean(train_loss)), end=True)

            self.runValidation(quick=False)
            self.scheduler.step()
            print(f'learning rate: {self.scheduler.get_last_lr()[0]}')

            #TODO
            self.model.saveModelToFile(self.logname + "/")

            if epoch % self.log_freq == 0 and self.instance_name is not None:
                self._uploadToCloud(epoch)

        if self.instance_name is not None:
            self._uploadToCloud()
    
    
    def runValidation(self, quick=False, pnt=True): 
        if not quick:
            print("Running full validation...")
        val_loss = []
        for step, (d_in, d_out) in enumerate(self.val_ds):
            val_loss.append(self.step(d_in, d_out, train=False))
            if quick:
                break
        #TODO
        '''d_in_graphs  = (tf.tile(tf.expand_dims(d_in[0][0], 0),[50,1]), tf.tile(tf.expand_dims(d_in[1][0], 0),[50,1,1]), tf.tile(tf.expand_dims(d_in[2][0], 0),[50,1,1]))
        d_out_graphs = (tf.tile(tf.expand_dims(d_out[0][0], 0),[50,1,1]), tf.tile(tf.expand_dims(d_out[1][0], 0),[50,1]), 
                        tf.tile(tf.expand_dims([d_out[2][0]], 0),[50,1]), tf.tile(tf.expand_dims(d_out[3][0], 0),[50,1,1]))
        self.createGraphs((d_in[0][0], d_in[1][0], d_in[2][0]),
                          (d_out[0][0], d_out[1][0], d_out[2][0], d_out[3][0]), 
                          self.model(d_in_graphs, training=True, use_dropout=True))'''
        if pnt:
            print("  Validation Loss: {:.6f}".format(np.mean(val_loss)))
        return np.mean(val_loss)

    def step(self, d_in, d_out, train, result):
        '''print('inside step: din')
        print(d_in[0].shape)
        print(d_in[1].shape)
        print(d_in[2].shape)'''
        #result = self.model(d_in, training=train)
        #tf_dout = [tf.convert_to_tensor(inpt.detach().cpu().numpy()) for inpt in d_out]
        #tf_result = [[tf.convert_to_tensor(inpt.detach().cpu().numpy()) for inpt in result_tp] for result_tp in result]
        #print(f'transformed result: {tf_result[0]}')
        #tf_loss, _ = self.tf_test_nw.calculateLoss(tf_dout, tf_result, train)
        #print(f'tf loss: {tf_loss} and dtype: {tf_loss.dtype}')
        loss, (atn, trj, dt, phs, wght) = self.calculateLoss(d_out, result, train)
        return loss

        #print(f'torch loss: {loss} and dtype: {loss.dtype}')

        if train:
            if not self.optimizer:
                self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999)) 
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=self.gamma_sl)
            #print(f'num parametrs in model: {len(list(self.model.parameters()))}')

            self.optimizer.zero_grad()
            h = time.perf_counter()

            loss.backward()

            '''for i, para in enumerate(list(self.model.parameters())):
                print(f'para num {i} has grad sum {para.grad.detach().sum()}')'''
            #print(f'gradient: {list(self.model.parameters())[0].grad}')
            self.optimizer.step()

            self.tboard.addTrainScalar("Loss", loss, self.global_step)
            self.tboard.addTrainScalar("Loss Attention", atn, self.global_step)
            self.tboard.addTrainScalar("Loss Trajectory", trj, self.global_step)
            self.tboard.addTrainScalar("Loss Phase", phs, self.global_step)
            self.tboard.addTrainScalar("Loss Weight", wght, self.global_step)
            self.tboard.addTrainScalar("Loss Delta T", dt, self.global_step)
        else:
            if self.last_written_step != self.global_step:
                self.last_written_step = self.global_step
                self.tboard.addValidationScalar("Loss", loss, self.global_step)
                self.tboard.addValidationScalar("Loss Attention", atn, self.global_step)
                self.tboard.addValidationScalar("Loss Trajectory", trj, self.global_step)
                self.tboard.addValidationScalar("Loss Phase", phs, self.global_step)
                self.tboard.addValidationScalar("Loss Weight", wght, self.global_step)
                self.tboard.addValidationScalar("Loss Delta T", dt, self.global_step)
                loss = loss.detach().cpu()
                if loss < self.global_best_loss:
                    self.global_best_loss = loss
                    self.model.saveModelToFile(self.logname + "/best/")
                    print(f'model saved with loss: {loss}')

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

    def catCrossEntrLoss(self, y_labels, y_preds):
        y_labels_args = torch.argmax(y_labels, dim = -1)

        return self.ce_loss(y_preds, y_labels_args)

    def calculateLoss(self, d_out, result, train):
        gen_trj, atn, dmp_dt, phs, wght                       = result
        generated, attention, delta_t, weights, phase, loss_atn = d_out

        weight_dim  = torch.tensor([3.0, 3.0, 3.0, 1.0, 0.5, 1.0, 0.1], device = generated.device)
        
        atn_loss = self.catCrossEntrLoss(attention, atn)
        #print(f'atn_loss: {atn_loss}')

        dt_loss = self.mse_loss(delta_t, dmp_dt[:,0]).mean()
        #print(f'dt_loss: {dt_loss}')

        repeated_weight_dim = weight_dim.reshape(1,1,-1).repeat([gen_trj.size(0), gen_trj.size(1), 1])
        trj_loss = self.calculateMSEWithPaddingMask(generated, gen_trj, repeated_weight_dim)
        trj_loss = (trj_loss * loss_atn).mean()
        #print(f'trj_loss: {trj_loss}')
   
        phs_loss = self.calculateMSEWithPaddingMask(phase, phs[:,:,0], loss_atn).mean()
        #print(f'phs_loss: {phs_loss}')
        #TODO why :-1?'''
        weight_loss = nn.MSELoss(reduction='none')(wght[:,:-1,:,:], wght[:,:-1,:,:].roll(shifts = -1, dims = 1)).mean((-2,-1))
        '''print('weight loss')
        print(weight_loss.shape)'''
        weight_loss = (weight_loss * loss_atn[:,:-1]).mean()
        #print(f'weight_loss: {weight_loss}')
        return (atn_loss * self.lw_atn +
                trj_loss * self.lw_trj +
                phs_loss * self.lw_phs +
                weight_loss * self.lw_w + 
                dt_loss  * self.lw_dt,
                (atn_loss, trj_loss, dt_loss, phs_loss, weight_loss)
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

    def createGraphs(self, d_in, d_out, result):
        language, image, robot_states            = d_in
        target_trj, attention, delta_t, weights  = d_out
        gen_trj, (atn, dmp_dt, phase, wght)      = result

        self.tboard.plotClassAccuracy(attention, tf.math.reduce_mean(atn, axis=0), tf.math.reduce_std(atn, axis=0), language, stepid=self.global_step)
        self.tboard.plotDMPTrajectory(target_trj, tf.math.reduce_mean(gen_trj, axis=0), tf.math.reduce_std(gen_trj, axis=0),
                                      tf.math.reduce_mean(phase, axis=0), delta_t, tf.math.reduce_mean(dmp_dt, axis=0), stepid=self.global_step)
