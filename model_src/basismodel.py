# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import numpy as np
#from model_src.basismodelTorch import BasisModelTorch

class BasisModel(tf.keras.layers.Layer):
    def __init__(self, dimensions, nfunctions, scale, **kwarg):
        super(BasisModel, self).__init__(name="attention", **kwarg)
        self._degree = nfunctions
        self.scale   = scale
        #self.basismodelTorch = BasisModelTorch(nfunctions = nfunctions, scale = scale)

    def build(self, input_shape):
        self.centers = np.linspace(0.0, 1.01, self._degree, dtype = np.float32)
        self.centers = tf.convert_to_tensor(self.centers)

    @tf.function
    def call(self, inputs, training=None):
        weights     = tf.transpose(inputs[0], perm=[0,2,1])
        weights_std = inputs[1]
        positions   = inputs[2]
        basis_funcs = self.compute_basis_values(positions)
        #print(f'shape basis func tf: {basis_funcs.shape}')
        #basis_funcs = tf.convert_to_tensor(self.basismodelTorch.compute_basis_values(torch.tensor(positions.numpy()).unsqueeze(1)).numpy())
        #print(f'shape basis func torch: {basis_funcs.shape}')
        result      = tf.linalg.matmul(basis_funcs, weights)
        '''print(f'weights shape: {weights.shape}')
        print(f'basis_funcs shape: {basis_funcs.shape}')
        print(f'result shape: {result.shape}')'''

        return result, tf.zeros_like(result)

    #def get_config(self):
    #   config = super(TopDownAttention, self).get_config()
    #    config.update({'units': self.units})
    #    return config

    def compute_basis_values(self, x):
        centers = tf.tile(tf.expand_dims(self.centers,0), [tf.shape(x)[1], 1])
        x       = tf.expand_dims(x, 2)
        funcs   = tf.exp(-( tf.math.pow((x - centers), 2) / (2.0 * self.scale) ))
        '''print(f'centers shape: {centers.shape}')
        print(f'x shape: {x.shape}')
        print(f'funcs shape: {funcs.shape}')'''
        return funcs
