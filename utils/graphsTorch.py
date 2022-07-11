# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from pickle import NONE
from urllib.parse import non_hierarchical
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from hashids import Hashids
from LanguagePolicies.utils.voice import Voice
import os
from MetaWorld.searchTest.toyEnvironment import make_sliding_tol



class TBoardGraphsTorch():
    def __init__(self, logname= None, data_path = None):
        if logname is not None:
            self.__hashids           = Hashids()
            #self.logdir              = "Data/TBoardLog/" + logname + "/"
            self.logdir              = os.path.join(data_path, "gboard/" + logname + "/")
            print(f'log dir: {self.logdir + "train/"}')
            self.__tboard_train      = tf.summary.create_file_writer(self.logdir + "train/")
            self.__tboard_validation = tf.summary.create_file_writer(self.logdir + "validate/")
            #self.voice               = Voice(path=data_path)
        self.fig, self.ax = plt.subplots(3,3)

    def startDebugger(self):
        tf.summary.trace_on(graph=True, profiler=True)
    
    def stopDebugger(self):
        with self.__tboard_validation.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=self.logdir)

    def finishFigure(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def addTrainScalar(self, name, value, stepid):
        with self.__tboard_train.as_default():
            tfvalue = self.torch2tf(value)
            tf.summary.scalar(name, tfvalue, step=stepid)

    def addValidationScalar(self, name, value, stepid):
        with self.__tboard_validation.as_default():
            tfvalue = self.torch2tf(value)
            tf.summary.scalar(name, tfvalue, step=stepid)

    def torch2tf(self, inpt):
        if inpt is not None:
            return tf.convert_to_tensor(inpt.detach().cpu().numpy())
        else:
            return inpt


    def plotDMPTrajectory(self, y_true, y_pred, y_pred_std = None, phase= None, \
        dt= None, p_dt= None, stepid= None, name = "Trajectory", save = False, \
            name_plot = None, path=None, tol_neg = None, tol_pos=None, inpt = None, opt_gen_trj=None, window = 0):
        tf_y_true = self.torch2tf(y_true)
        tf_y_pred = self.torch2tf(y_pred)
        tf_phase = self.torch2tf(phase)
        tf_inpt = self.torch2tf(inpt)
        if p_dt is not None:
            tf_dt = self.torch2tf(dt)
            tf_p_dt = self.torch2tf(p_dt)
        if opt_gen_trj is not None:
            tf_opt_gen_trj = self.torch2tf(opt_gen_trj)
            tf_opt_gen_trj = tf_opt_gen_trj.numpy()

        tf_y_true      = tf_y_true.numpy()
        tf_y_pred      = tf_y_pred.numpy()
        tf_inpt        = tf_inpt.numpy()
        if tf_phase is not None:
            tf_phase       = tf_phase.numpy()

        if p_dt is not None:
            tf_dt          = tf_dt.numpy() * 350.0
            tf_p_dt        = tf_p_dt.numpy()
        trj_len      = tf_y_true.shape[0]
        
        #fig, ax = plt.subplots(3,3)
        fig, ax = self.fig, self.ax
        #fig.set_size_inches(9, 9)
        if tol_neg is not None:
            if window > 0:
                neg_inpt, pos_inpt, tf_y_true = make_sliding_tol(label=y_true.unsqueeze(0), neg_tol=tol_neg, pos_tol=tol_pos, window=window)
                tf_y_true = tf_y_true[0].detach().cpu().numpy()
                trj_len      = tf_y_true.shape[0]
                tf_y_pred = y_pred[int(window/2):-(int(window/2) + 1)].detach().cpu().numpy()
                neg_inpt = neg_inpt.detach().cpu().numpy()
                pos_inpt = pos_inpt.detach().cpu().numpy()
                if opt_gen_trj is not None:
                    tf_opt_gen_trj = opt_gen_trj[int(window/2):-(int(window/2) + 1)].detach().cpu().numpy()
            else:
                neg_inpt = tf_y_true + tol_neg[None,:].cpu().numpy()
                pos_inpt = tf_y_true + tol_pos[None,:].cpu().numpy()
        for sp in range(len(tf_y_true[0])):
            idx = sp // 3
            idy = sp  % 3
            ax[idx,idy].clear()

            # GT Trajectory:
            if tol_neg is not None:
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), neg_inpt[:,sp], alpha=0.75, color='orangered')
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), pos_inpt[:,sp], alpha=0.75, color='orangered')
            ax[idx,idy].plot(range(trj_len), tf_y_true[:,sp],   alpha=1.0, color='forestgreen')            
            ax[idx,idy].plot(range(tf_y_pred.shape[0]), tf_y_pred[:,sp], alpha=0.75, color='mediumslateblue')
            if opt_gen_trj is not None:
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), tf_opt_gen_trj[:,sp], alpha=0.75, color='lightseagreen')
                diff_vec = tf_opt_gen_trj - tf_y_pred
                ax[idx,idy].plot(range(tf_y_pred.shape[0]), diff_vec[:,sp], alpha=0.75, color='pink')

            #ax[idx,idy].errorbar(range(tf_y_pred.shape[0]), tf_y_pred[:,sp], xerr=None, yerr=None, alpha=0.25, fmt='none', color='mediumslateblue')
            #ax[idx,idy].set_ylim([-0.1, 1.1])
            if p_dt is not None:
                ax[idx,idy].plot([tf_dt, tf_dt], [0.0,1.0], linestyle=":", color='forestgreen')

        if inpt is not None:
            ax[-1,-1].clear()
            ax[-1,-1].plot(range(inpt.shape[-1]), tf_inpt,   alpha=1.0, color='forestgreen')     
        
        if tf_phase is not None:
            ax[2,2].clear()
            ax[2,2].plot(range(tf_y_pred.shape[0]), tf_phase, color='orange')
        if p_dt is not None:
            ax[2,2].plot([tf_dt, tf_dt], [0.0,1.0], linestyle=":", color='forestgreen')
            ax[2,2].plot([tf_p_dt*350.0, tf_p_dt*350.0], [0.0,1.0], linestyle=":", color='mediumslateblue')
            ax[2,2].set_ylim([-0.1, 1.1])

        result = np.expand_dims(self.finishFigure(fig), 0)
        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name_plot + '.png')
        #fig.clear()
        #plt.close()
        if not save:
            with self.__tboard_validation.as_default():
                tf.summary.image(name, data=result, step=stepid)