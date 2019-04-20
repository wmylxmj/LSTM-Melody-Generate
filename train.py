# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:49:35 2019

@author: wmy
"""

import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Input, LSTM, Reshape, Lambda, RepeatVector
from keras.optimizers import Adam
from keras import backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from utils import DataLoader
from model import MidiGenerateModel

# load dataset 
data_loader = DataLoader()
X, Y = data_loader.load_dataset('./datasets/midi', Tx=64, sample_interval=4)
m, Tx, n_values = X.shape
Y = Y.reshape((Tx, m, n_values))
print('[OK] dataset loaded, find {} samples.'.format(m))

# creat the model
midi_model = MidiGenerateModel(n_values)
model = midi_model.train_model(Tx)
print('[OK] model created.')

# compile the model
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print('[OK] model compiled.')

# init a0, c0
a0_1 = np.zeros((m, midi_model.lstm_layer_1_units))
c0_1 = np.zeros((m, midi_model.lstm_layer_1_units))
a0_2 = np.zeros((m, midi_model.lstm_layer_2_units))
c0_2 = np.zeros((m, midi_model.lstm_layer_2_units))

# keras callback
tensorboard = TensorBoard(log_dir='./logs/train_1', 
                          write_graph=True, 
                          write_images=True)

checkpoint = ModelCheckpoint(filepath='./weights/weights.h5', 
                             monitor='loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto',
                             period=1)

callback_lists = [tensorboard, checkpoint]
print('[OK] callbacks created.')

# train
history = model.fit([X, a0_1, c0_1, a0_2, c0_2], list(Y), epochs=10000, \
                    batch_size=512, shuffle=True, callbacks=callback_lists)

