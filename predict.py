# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:39:27 2019

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
from parsers import MidiParser, SequenceParser
from model import MidiGenerateModel
from utils import DataLoader

midi_parser = MidiParser()
sequence_parser = SequenceParser()
n_values = (midi_parser.pitch_span + 1) * len(sequence_parser.durations)
midi_model = MidiGenerateModel(n_values)

model = midi_model.predict_model()
model.load_weights('./weights/weights.h5')

x_initializer = np.zeros((1, 1, n_values))
a_1_initializer = np.zeros((1, midi_model.lstm_layer_1_units))
c_1_initializer = np.zeros((1, midi_model.lstm_layer_1_units))
a_2_initializer = np.zeros((1, midi_model.lstm_layer_2_units))
c_2_initializer = np.zeros((1, midi_model.lstm_layer_2_units))

x = x_initializer
a_1 = a_1_initializer
c_1 = c_1_initializer
a_2 = a_2_initializer
c_2 = c_2_initializer
    
matrix = []
for i in tqdm(range(256)):
    x, a_1, c_1, a_2, c_2 = model.predict([x, a_1, c_1, a_2, c_2])
    matrix.append(x[0][0])
    pass

matrix = np.array(matrix)
sequence = sequence_parser.matrix_to_monosyllabic_melody(matrix)
sequence = np.array(sequence)
midi_parser.unparse(sequence, './outputs/generate.mid')
