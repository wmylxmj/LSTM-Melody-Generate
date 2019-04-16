# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:09:42 2019

@author: wmy
"""

import midi
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from parsers import MidiParser, SequenceParser

class MidiClipper(object):
    '''Midi Clipper'''
    
    def __init__(self, name=None):
        self.midi_parser = MidiParser()
        self.sequence_parser = SequenceParser()
        self.name = name
        pass
    
    def clip(self, fp, tracks=None, offsets=None):
        sequence = self.midi_parser.parse(fp, tracks)
        if offsets == None:
            return sequence
        elif not isinstance(offsets, tuple):
            raise ValueError("offsets must be tuple")
        if offsets[0] == None:
            offsets = (0, offsets[1])
            pass
        if offsets[1] == None:
            return sequence[offsets[0]:-1]
        return sequence[offsets[0]:offsets[1]]
    
    def collage(self, sequences, sp):
        if not isinstance(sequences, list):
            raise ValueError("sequences must be list")
        collaged_sequence = sequences[0]
        if len(sequences) == 1:
            self.midi_parser.unparse(collaged_sequence, sp)
            pass
        else:
            for sequence in sequences[1:]:
                collaged_sequence = np.concatenate([collaged_sequence, sequence], axis=0)
                pass
            self.midi_parser.unparse(collaged_sequence, sp)
            pass
        pass
    
    pass
