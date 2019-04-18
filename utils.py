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

class DatasetsLoader(object):
    '''Datasets Loader'''
    
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
    
    def load_datasets(self, folder, Tx, sample_interval=8):
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


class DatasetLoader(object):
    '''Dataset Loader'''
    
    def __init__(self, name=None):
        self.midi_parser = MidiParser()
        self.sequence_parser = SequenceParser()
        self.name = name
        pass
    
    def midi_to_matrix(self, fp):
        sequence = self.midi_parser.parse(fp)
        span = sequence.shape[1]
        melody = self.sequence_parser.get_monosyllabic_melody(sequence)
        notes = self.sequence_parser.get_notes(melody)
        notes = self.sequence_parser.sort_notes(notes, mode=1, reverse=False)
        notes_set = []
        last_end_time = notes[0][1]
        for note in notes:
            if note[1] != last_end_time:
                rest_time = note[1] - last_end_time
                notes_set.append(str(span) + '_' + str(rest_time))
                pass
            note_time = note[2] - note[1]
            notes_set.append(str(note[0]) + '_' + str(note_time))
            last_end_time = note[2]
            pass
        notes_set = set(notes_set)
        # sort notes set
        sorted_notes_set = []
        for note in notes_set:
            sorted_notes_set.append(note)
            pass
        sorted_notes_set = sorted(sorted_notes_set)  
        # get dicts
        note_to_index_dict = {}
        index_to_note_dict = {}
        for i, note in enumerate(sorted_notes_set):
            note_to_index_dict[note] = str(i)
            index_to_note_dict[str(i)] = note
            pass
        len_vector = len(note_to_index_dict)
        matrix = []
        last_end_time = notes[0][1]
        for note in notes:
            if note[1] != last_end_time:
                rest_time = note[1] - last_end_time
                note_info = str(span) + '_' + str(rest_time)
                index = int(note_to_index_dict[note_info])
                vector = [0 for x in range(len_vector)]
                vector[index] = 1
                matrix.append(vector)
                pass
            note_time = note[2] - note[1]
            note_info = str(note[0]) + '_' + str(note_time)
            index = int(note_to_index_dict[note_info])
            vector = [0 for x in range(len_vector)]
            vector[index] = 1
            matrix.append(vector)
            last_end_time = note[2]
            pass
        matrix = np.array(matrix)
        return matrix, index_to_note_dict
    
    def load_dataset(self, fp, Tx, sample_interval=8):
        matrix, index_to_note_dict = self.midi_to_matrix(fp)
        matrix = np.array(matrix)
        X, Y = [], []
        for i in range(0, matrix.shape[0]-Tx-1, sample_interval):
            X.append(matrix[i:i+Tx])
            Y.append(matrix[i+1:i+Tx+1])
            pass
        X = np.array(X)
        Y = np.array(Y)
        return X, Y, index_to_note_dict
    
    def matrix_to_midi(self, matrix, index_to_note_dict, sp):
        sequence = []
        span = self.midi_parser.pitch_span
        for i in range(matrix.shape[0]):
            vector = matrix[i]
            index = np.argmax(vector)
            note = index_to_note_dict[str(index)]
            pitch, time = note.split('_')
            pitch = int(pitch)
            time = int(time)
            if pitch == span:
                state = [0 for x in range(span)]
                for t in range(time):
                    sequence.append(state)
                    pass
                pass
            else:
                state = [0 for x in range(span)]
                state[pitch] = 1
                for t in range(time):
                    sequence.append(state)
                    pass
                pass
            pass
        sequence = np.array(sequence)
        self.midi_parser.unparse(sequence, sp)
        pass
    
    pass
        

class DataLoader(object):
    '''Data Loader'''
    
    def __init__(self, name=None):
        self.datasets_loader = DatasetsLoader()
        self.dataset_loader = DatasetLoader()
        self.name = name
        pass
    
    def load_dataset(self, fp, Tx, sample_interval=8):
        return self.dataset_loader.load_dataset(fp, Tx, sample_interval)
    
    def load_datasets(self, folder, Tx, sample_interval=8):
        return self.datasets_loader.load_datasets(folder, Tx, sample_interval)
    
    pass
        
        