# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:23:21 2019

@author: wmy
"""

import midi
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Input, LSTM, Reshape, Lambda, RepeatVector
from keras.optimizers import Adam
from keras import backend as K
from IPython.display import SVG
import tensorflow as tf
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

class MidiGenerateModel(object):
    
    def __init__(self, n_values):
        self.lstm_layer_1_units = 256
        self.lstm_layer_2_units = 256
        self.dense_1_units = 512
        self.dense_2_units = n_values
        self.n_values = n_values
        pass
    
    def train_model(self, Tx):
        lstm_layer_1 = LSTM(self.lstm_layer_1_units, return_state=True) 
        lstm_layer_2 = LSTM(self.lstm_layer_2_units, return_state=True) 
        dense_layer_1 = Dense(self.dense_1_units, activation='relu')
        dense_layer_2 = Dense(self.dense_2_units, activation='softmax')
        X = Input(shape=(Tx, self.n_values))
        a0_1 = Input(shape=(self.lstm_layer_1_units, ), name='a0_1')
        c0_1 = Input(shape=(self.lstm_layer_1_units, ), name='c0_1')
        a0_2 = Input(shape=(self.lstm_layer_2_units, ), name='a0_2')
        c0_2 = Input(shape=(self.lstm_layer_2_units, ), name='c0_2')
        a_1 = a0_1
        c_1 = c0_1
        a_2 = a0_2
        c_2 = c0_2
        outputs = []
        for t in range(Tx):
            x = Lambda(lambda x: X[:, t, :])(X)
            x = Reshape((1, self.n_values))(x)
            a_1, _, c_1 = lstm_layer_1(x, initial_state=[a_1, c_1])
            a_next = Reshape((1, self.lstm_layer_1_units))(a_1)
            a_2, _, c_2 = lstm_layer_2(a_next, initial_state=[a_2, c_2])
            out = dense_layer_1(a_2)
            out = dense_layer_2(out)
            outputs.append(out)
            pass
        model = Model([X, a0_1, c0_1, a0_2, c0_2], outputs)
        return model
    
    def predict_model(self):
        def one_hot(x):
            x = K.argmax(x)
            x = tf.one_hot(x, self.n_values) 
            x = RepeatVector(1)(x)
            return x
        lstm_layer_1 = LSTM(self.lstm_layer_1_units, return_state=True) 
        lstm_layer_2 = LSTM(self.lstm_layer_2_units, return_state=True) 
        dense_layer_1 = Dense(self.dense_1_units, activation='relu')
        dense_layer_2 = Dense(self.dense_2_units, activation='softmax')
        x0 = Input(shape=(1, self.n_values))
        a0_1 = Input(shape=(self.lstm_layer_1_units, ), name='a0_1')
        c0_1 = Input(shape=(self.lstm_layer_1_units, ), name='c0_1')
        a0_2 = Input(shape=(self.lstm_layer_2_units, ), name='a0_2')
        c0_2 = Input(shape=(self.lstm_layer_2_units, ), name='c0_2')
        a_1, _, c_1 = lstm_layer_1(x0, initial_state=[a0_1, c0_1])
        a_next = Reshape((1, self.lstm_layer_1_units))(a_1)
        a_2, _, c_2 = lstm_layer_2(a_next, initial_state=[a0_2, c0_2])
        out = dense_layer_1(a_2)
        out = dense_layer_2(out)
        x = Lambda(one_hot)(out)
        model = Model([x0, a0_1, c0_1, a0_2, c0_2], [x, a_1, c_1, a_2, c_2])
        return model
    
    pass
            
            
        