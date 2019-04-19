# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:27:45 2019
@author: wmy
"""

import midi
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from parsers import MidiParser, SequenceParser

class DataLoader(object):
    '''Data Loader'''
    
    def __init__(self, name=None):
        self.midi_parser = MidiParser()
        self.sequence_parser = SequenceParser()
        self.name = name
        pass
    
    def load_songs(self, folder):
        files = glob.glob('{}/*.mid*'.format(folder))
        songs = []
        for file in files:
            song = self.midi_parser.parse(file)
            song = np.array(song)
            songs.append(song)
            pass
        return songs
    
    def load_dataset(self, folder, Tx, sample_interval=8):
        songs = self.load_songs(folder)
        X, Y = [], []
        for song in tqdm(songs):
            melody = self.sequence_parser.get_monosyllabic_melody(song)
            melody = self.sequence_parser.monosyllabic_melody_to_matrix(melody)
            for i in range(0, melody.shape[0]-Tx-1, sample_interval):
                X.append(melody[i:i+Tx])
                Y.append(melody[i+1:i+Tx+1])
                pass
            pass
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    
    pass

