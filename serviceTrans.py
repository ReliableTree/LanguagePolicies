#!/usr/bin/env python

# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

from torch._C import dtype
import rclpy
from policy_translation.srv import NetworkPT, TuneNetwork
from model_src.modelTorch import PolicyTranslationModelTorch
from utils.tf_util import trainOnCPU, limitGPUMemory
from utils.safe_cfeatures import save_dict_of_features
import tensorflow as tf
import numpy as np
import re
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt
from utils.intprim.gaussian_model import GaussianModel
import pickle
import torch
import torch.nn as nn
import time
# Force TensorFlow to use the CPU
FORCE_CPU    = True
# Use dropout at run-time for stochastif-forward passes
USE_DROPOUT  = True
# Where can we find the trained model?
# Where is a pre-trained faster-rcnn?
FRCNN_PATH   = "/home/hendrik/Documents/master_project/LokalData/GDrive/rcnn"
# Where are the GloVe word embeddings?
GLOVE_PATH   = "/home/hendrik/Documents/master_project/LokalData/GDrive/glove.6B.50d.txt"
# Where is the normalization of the dataset?
NORM_PATH    = "/home/hendrik/Documents/master_project/LokalData/GDrive/normalization_v2.pkl"


# Location of the training data
TRAIN_DATA      = "/home/hendrik/Documents/master_project/LokalData/GDrive/train.tfrecord"
# Location of the validation data
VALIDATION_DATA = "/home/hendrik/Documents/master_project/LokalData/GDrive/validate.tfrecord"
# Location of the GloVe word embeddings
GLOVE_PATH      = "/home/hendrik/Documents/master_project/LokalData/GDrive/glove.6B.50d.txt"
# Learning rate for the adam optimizer
LEARNING_RATE   = 1e-4
# Weight for the attention loss
WEIGHT_ATTN     = 1.0
# Weight for the motion primitive weight loss
WEIGHT_W        = 50.0
# Weight for the trajectroy generation loss
WEIGHT_TRJ      = 5.0
# Weight for the time progression loss
WEIGHT_DT       = 14.0
# Weight for the phase prediction loss
WEIGHT_PHS      = 1.0
# Number of epochs to train
TRAIN_EPOCHS    = 1

TRAIN_DATA_TORCH = '/home/hendrik/Documents/master_project/LokalData/TorchDataset/train_data_torch.txt'

VAL_DATA_TORCH = '/home/hendrik/Documents/master_project/LokalData/TorchDataset/val_data_torch.txt'

MODEL_PATH   = '/home/hendrik/Documents/master_project/LokalData/Model/NxmJWy8Egzz/best_val/policy_translation_h'
MODEL_SETUP  = '/home/hendrik/Documents/master_project/LokalData/Model/NxmJWy8Egzz/best_val/model_setup.pkl'
MEMORY_PATH  = '/home/hendrik/Documents/master_project/LokalData/Model/NxmJWy8Egzz/best_val/'


from torch.utils.data import DataLoader
from utils.convertTFDataToPytorchData import TorchDataset
from utils.networkTorch import NetworkTorch
DEVICE = 'cpu'

if FORCE_CPU:
    trainOnCPU()
else:
    limitGPUMemory()

# should not use network.savedict.....
def setup_model(device = 'cpu', batch_size = 32):
    with open(MODEL_SETUP, 'rb') as f:
        model_setup = pickle.load(f)
        print(model_setup)
    model_setup['use_memory'] = True
    model_setup['train'] = False
    model_setup['use_dropout'] = True
    model   = PolicyTranslationModelTorch(od_path=FRCNN_PATH, glove_path=GLOVE_PATH, model_setup=model_setup).to(device)
    train_data = TorchDataset(path = TRAIN_DATA_TORCH, device=device, on_device=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if 'use_memory' in model_setup and model_setup['use_memory']:
        print('use memory')
        model.load_memory(MEMORY_PATH, ['start_position', 'cfeatures'])
    eval_data = TorchDataset(path = VAL_DATA_TORCH, device=device)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
    network = NetworkTorch(model, data_path='/home/hendrik/Documents/master_project/LokalData', logname='LOGNAME', lr=LEARNING_RATE, lw_gen_trj = 0, lw_fod = 0,  lw_atn=WEIGHT_ATTN, lw_w=WEIGHT_W, lw_trj=WEIGHT_TRJ, lw_dt=WEIGHT_DT, lw_phs=WEIGHT_PHS)
    network.setDatasets(train_loader=train_loader, val_loader=eval_loader)
    network.setup_model(model_params=model_setup)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda:0'))

    return model

class NetworkService():
    def __init__(self, model, device = 'cpu'):
        self.dictionary    = self._loadDictionary(GLOVE_PATH)
        self.regex         = re.compile('[^a-z ]')
        self.bridge        = CvBridge()
        self.history       = []
        rclpy.init(args=None)
        self.model = model
        self.device = device
        self.node = rclpy.create_node("neural_network")
        self.service_nn = self.node.create_service(NetworkPT,   "/network",      self.cbk_network_dmp_ros2)
        self.normalization = pickle.load(open(NORM_PATH, mode="rb"), encoding="latin1")
        self.last_GRU_state = None
        with open(MODEL_SETUP, 'rb') as f:
            self.model_setup = pickle.load(f)
            self.model_setup['train'] = True
        self.trj_gen = None
        print("Ready")

    def runNode(self):
        while rclpy.ok():
            rclpy.spin_once(self.node)
        self.node.destroy_service(self.service_nn)
        self.node.destroy_service(self.service_tn)
        rclpy.shutdown()

    def _loadDictionary(self, file):
        __dictionary = {}
        __dictionary[""] = 0 # Empty string
        fh = open(file, "r", encoding="utf-8")
        for line in fh:
            if len(__dictionary) >= 300000:
                break
            tokens = line.strip().split(" ")
            __dictionary[tokens[0]] = len(__dictionary)
        fh.close()
        return __dictionary

    def tokenize(self, language):
        voice  = self.regex.sub("", language.strip().lower())
        tokens = []
        for w in voice.split(" "):
            idx = 0
            try:
                idx = self.dictionary[w]
            except:
                print("Unknown word: " + w)
            tokens.append(idx)
        return tokens
    
    def normalize(self, value, v_min, v_max):
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            raise ArrayDimensionMismatch()
        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = (value - v_min) / (v_max - v_min)
        return value

    def interpolateTrajectory(self, trj, target):
        current_length = trj.shape[0]
        dimensions     = trj.shape[1]
        result         = np.zeros((target, trj.shape[1]), dtype=np.float32)
              
        for i in range(dimensions):
            result[:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[:,i])
        
        return result

    def cbk_network_dmp_ros2(self, req, res):
        res.trajectory, res.confidence, res.timesteps, res.weights, res.phase = self.cbk_network_dmp(req)
        return res
    
    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):   
        if img_msg.encoding != "8UC3":     
            self.node.get_logger().info("Unrecognized image type: " + encoding)
            exit(0)
        dtype      = "uint8"
        n_channels = 3

        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

        img_buf = np.asarray(img_msg.data, dtype=dtype) if isinstance(img_msg.data, list) else img_msg.data

        if n_channels == 1:
            im = np.ndarray(shape=(img_msg.height, img_msg.width),
                            dtype=dtype, buffer=img_buf)
        else:
            im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                            dtype=dtype, buffer=img_buf)
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder()

        if desired_encoding == 'passthrough':
            return im

        from cv_bridge.boost.cv_bridge_boost import cvtColor2

        try:
            res = cvtColor2(im, img_msg.encoding, desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def cbk_network_dmp(self, req):
        if req.reset:
            self.req_step = 0
            self.sfp_history = []
            self.trj_gen = None
            try:
                image = self.imgmsg_to_cv2(req.image)
            except CvBridgeError as e:
                print(e)
            language = self.tokenize(req.language)
            self.language = language + [0] * (15-len(language))

            image_features = self.model.frcnn(tf.convert_to_tensor([image], dtype=tf.uint8))
            #print(image_features["detection_scores"].shape)
            scores   = image_features["detection_scores"][0, :6].numpy().astype(dtype=np.float32)
            scores   = [0.0 if v < 0.5 else 1.0 for v in scores.tolist()]

            classes  = image_features["detection_classes"][0, :6].numpy().astype(dtype=np.int32)
            classes  = [v * scores[k] for k, v in enumerate(classes.tolist())]

            boxes    = image_features["detection_boxes"][0, :6, :].numpy().astype(dtype=np.float32)
            
            self.features = np.concatenate((np.expand_dims(classes,1), boxes), axis=1)

            self.history  = []   
            self.last_GRU_state = None     

        self.history.append(list(req.robot)) 

        robot           = np.asarray(self.history, dtype=np.float32)
        #robot           = np.asarray([list(req.robot)], dtype=np.float32)
        do_dim = 100
        self.input_data = (
            torch.tensor(np.tile([self.language],[do_dim, 1]), dtype=torch.int64, device=self.device), 
            torch.tensor(np.tile([self.features],[do_dim, 1, 1]), dtype=torch.float32, device=self.device),
            torch.tensor(np.tile([robot],[do_dim, 1, 1]), dtype=torch.float32, device=self.device),
            self.last_GRU_state
        )
        #HENDRIK
        h = time.perf_counter()
        if self.trj_gen is None:
            if self.model.model_setup['use_dropout']:
                self.model.train()
            else:
                self.model.eval()
            with torch.no_grad():
                self.model.train()
                if 'predictionNN' in self.model_setup['contr_trans'] and self.model_setup['contr_trans']['predictionNN']:
                    generated, self.atn, phase, pred_loss, self.diff = self.model(self.input_data)
                else:
                    generated, self.atn, phase, self.diff = self.model(self.input_data)
            #print(f'time for one call: {time.perf_counter() - h}')
            #print(f'atn : {atn}')
            self.trj_gen    = generated.mean(axis=0)
            self.trj_gen    = self.median_fiter(self.trj_gen, 3)
            self.trj_gen = self.trj_gen.cpu().numpy() #350
            #print(f'trj_gen: {self.trj_gen.shape}')

            self.phase_value     = phase.mean(axis=0).unsqueeze(1)
            self.phase_value     = self.median_fiter(self.phase_value, 3)
            self.phase_value     = self.phase_value.cpu().numpy()

            self.pred_loss = pred_loss

            
        #print(f'phase: {phase_value}')
        res_trj = self.trj_gen[:min(349, len(self.history))]
        phase_value = self.phase_value[min(349, len(self.history))]
        '''if (phase_value > 0.95):# and len(self.sfp_history) > 100:
            #print(f'phase value: {phase_value}')
            np.save("history", self.history)

            gen_trajectory = []
            gen_trajectory = np.asarray(gen_trajectory)
            np.save("gen_trajectory", gen_trajectory)     
            dict_of_features = {
                'prompt' : str(req.language),
                #'cfeatures' : cfeatures[0].cpu().numpy(),
                'attn' : self.atn.cpu().numpy(),
                'features' : self.features
            }
            save_dict_of_features(dict_of_features, str(req.language))'''

            #save_cfeature(cfeatures[0].numpy(), str(req.language))
        
        self.req_step += 1
        return (res_trj.flatten().tolist(), [float(self.pred_loss.detach().mean())], 0, [float(self.diff)], float(phase_value)) 

    
    def median_fiter(self, inpt, median_size):
        result = None
        for i in range(inpt.size(0) - median_size + 1):
            if result is None:
                result = torch.median(inpt[i:i+median_size], dim = 0)[0].unsqueeze(0)
            else:
                result = torch.cat((result, torch.median(inpt[i:i+median_size], dim=0)[0].unsqueeze(0)), dim = 0)
        return result     
    
    
    def idToText(self, id):
        names = ["", "Yellow Small Round", "Red Small Round", "Green Small Round", "Blue Small Round", "Pink Small Round",
                     "Yellow Large Round", "Red Large Round", "Green Large Round", "Blue Large Round", "Pink Large Round",
                     "Yellow Small Square", "Red Small Square", "Green Small Square", "Blue Small Square", "Pink Small Square",
                     "Yellow Large Square", "Red Large Square", "Green Large Square", "Blue Large Square", "Pink Large Square",
                     "Cup Red", "Cup Green", "Cup Blue"]
        return names[id]
    
    def plotTrajectory(self, trj, error, image):
        fig, ax = plt.subplots(3,3)
        fig.set_size_inches(9, 9)

        for sp in range(7):
            idx = sp // 3
            idy = sp  % 3
            ax[idx,idy].clear()
            ax[idx,idy].plot(range(trj.shape[0]), trj[:,sp], alpha=0.5, color='mediumslateblue')
            ax[idx,idy].errorbar(range(trj.shape[0]), trj[:,sp], xerr=None, yerr=error[:,sp], alpha=0.1, fmt='none', color='mediumslateblue')
            ax[idx,idy].set_ylim([-0.1, 1.1])

        ax[2,1].imshow(image)

    def plotImageRegions(self, image_np, image_dict, atn):
        # Visualization of the results of a detection.
        tgt_object   = np.argmax(atn)
        num_detected = len([v for v in image_dict["detection_scores"][0] if v > 0.5]) 
        num_detected = min(num_detected, len(atn))
        for i in range(num_detected):
            ymin, xmin, ymax, xmax = image_dict['detection_boxes'][0][i,:]
            pt1 = (int(xmin*image_np.shape[1]), int(ymin*image_np.shape[0]))
            pt2 = (int(xmax*image_np.shape[1]), int(ymax*image_np.shape[0]))
            image_np = cv2.rectangle(image_np, pt1, pt2, (156, 2, 2), 1)
            if i == tgt_object:
                image_np = cv2.rectangle(image_np, pt1, pt2, (30, 156, 2), 2)
                image_np = cv2.putText(image_np, "{:.1f}%".format(atn[i] * 100), (pt1[0]-10, pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 156, 2), 2, cv2.LINE_AA)
            
        fig = plt.figure()
        plt.imshow(image_np)
    
if __name__ == "__main__":
    model = setup_model(device = 'cuda')
    ot = NetworkService(model, device = 'cuda')
    ot.runNode()