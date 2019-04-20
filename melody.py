# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:19:46 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from clipper import MidiClipper
from parsers import MidiParser, SequenceParser

midi_cliper = MidiClipper()
midi_parser = MidiParser()
sequence_parser = SequenceParser()

midi_folder = "./datasets/midi"
save_folder = "./datasets/melody"

files = os.listdir(midi_folder)
for file in files:
    _, ext = os.path.splitext(file.lower())
    if ext == ".mid" or ext == ".midi":
        sequence = midi_parser.parse(os.path.join(midi_folder, file))
        melody = sequence_parser.get_monosyllabic_melody(sequence)
        midi_parser.unparse(melody, os.path.join(save_folder, file))
        print('[OK] {} saved in {}.'.format(file, save_folder))
        pass
    pass
