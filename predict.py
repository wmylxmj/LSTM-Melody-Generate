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

np.random.seed(5)
x_initializer = np.random.rand(1, 1, n_values)
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

# generate companion notes
data_loader = DataLoader()
songs = data_loader.load_songs('./datasets/midi')
companion_matrixs = []
for song in tqdm(songs):
    companion_matrix = sequence_parser.get_companion_matrix(song)
    companion_matrixs.append(companion_matrix)
    pass
companion_matrix = companion_matrixs[0]
for cp_matrix in companion_matrixs[1:]:
    companion_matrix = np.add(companion_matrix, cp_matrix)
    pass

plt.rcParams['figure.dpi'] = 150
plt.imshow(companion_matrix)
plt.show()
plt.rcParams['figure.dpi'] = 100

sequence = midi_parser.parse('./outputs/generate.mid')
new = []
last_cp_index = None
notes = sequence_parser.get_notes(sequence)
notes = sequence_parser.sort_notes(notes)
last_end_time = notes[0][1]
for note in notes:
    note_pitch = note[0]
    time = note[2] - note[1]
    state = [0 for x in range(midi_parser.pitch_span)]
    if note[1] != last_end_time:
        rest_time = note[1] - last_end_time
        if last_cp_index != None:
            state[last_cp_index] = 1
            new.append(state)
            pass
        pass
    state = [0 for x in range(midi_parser.pitch_span)]
    state[note_pitch] = 1
    cps = companion_matrix[note_pitch]
    cp_index = np.argmax(cps)
    if time >= 8:
        state[cp_index] = 1
        last_cp_index = cp_index
        cps[cp_index] = 0
        if np.sum(cps) != 0:
            cp_index = np.argmax(cps)
            state[cp_index] = 1
            pass
        for i in range(time):
            new.append(state)
            pass
        pass
    elif time >= 4:
        state[cp_index] = 1
        last_cp_index = cp_index
        for i in range(time):
            new.append(state)
            pass
        pass
    else:
        if last_cp_index!=None:
            state[last_cp_index] = 1
        for i in range(time):
            new.append(state)
            pass
        pass
    
new = np.array(new)
midi_parser.unparse(new, './outputs/generate.mid')

