# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:52:39 2019

@author: wmy
"""

import midi
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class MidiParser(object):
    '''Midi Parser'''
    
    def __init__(self, name=None):
        self.__lowest_pitch = 21
        self.__highest_pitch = 108
        self.__pitch_span = self.__highest_pitch - self.__lowest_pitch + 1
        self.name = name
        pass
    
    @property
    def lowest_pitch(self):
        return self.__lowest_pitch
    
    @lowest_pitch.setter
    def lowest_pitch(self, pitch):
        if isinstance(pitch, int):
            if pitch >= 0:
                if pitch <= self.__highest_pitch:
                    self.__lowest_pitch = pitch
                    self.__pitch_span = self.__highest_pitch - self.__lowest_pitch + 1
                else:
                    raise ValueError("lowest pitch must be lower than highest pitch")
            else:
                raise ValueError("expected lowest pitch >= 0")
        else:
            raise ValueError("lowest pitch must be the type of int")
    
    @property
    def highest_pitch(self):
        return self.__highest_pitch
    
    @highest_pitch.setter
    def highest_pitch(self, pitch):
        if isinstance(pitch, int):
            if pitch <= 127:
                if pitch >= self.__lowest_pitch:
                    self.__highest_pitch = pitch
                    self.__pitch_span = self.__highest_pitch - self.__lowest_pitch + 1
                else:
                    raise ValueError("highest pitch must be higher than lowest pitch")
            else:
                raise ValueError("expected highest pitch <= 127")
        else:
            raise ValueError("highest pitch must be the type of int")
            
    @property
    def pitch_span(self):
        return self.__pitch_span
    
    def parse(self, fp, tracks=None):
        pattern = midi.read_midifile(fp)
        if tracks != None:
            if not isinstance(tracks, list):
                raise ValueError("tracks must be a list.")
            new_pattern = midi.Pattern()
            new_pattern.resolution = 480
            for index in tracks:
                if not isinstance(index, int):
                    raise ValueError("element in tracks must be int.")
                new_pattern.append(pattern[index])
                pass
            pattern = new_pattern
            pass
        sequence = []
        state = [0 for x in range(self.__pitch_span)]
        sequence.append(state)
        time_left = [track[0].tick for track in pattern]
        posns = [0 for track in pattern]
        time = 0
        while True:
            # duration: 1/64
            if time % (pattern.resolution//16) == (pattern.resolution//32):
                last_state = state
                state = [last_state[x] for x in range(self.__pitch_span)]
                sequence.append(state)
                pass
            for i in range(len(time_left)):
                while time_left[i] == 0:
                    track = pattern[i]
                    pos = posns[i]
                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch >= self.__lowest_pitch) or (evt.pitch <= self.__highest_pitch):
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-self.__lowest_pitch] = 0                      
                            else:
                                state[evt.pitch-self.__lowest_pitch] = 1   
                        pass
                    try:
                        time_left[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        time_left[i] = None
                    pass
                if time_left[i] is not None:
                    time_left[i] -= 1
            if all(t is None for t in time_left):
                break
            time += 1
            pass
        sequence = np.array(sequence)
        return sequence
    
    def unparse(self, sequence, sp):
        sequence = np.array(sequence)
        pattern = midi.Pattern()
        pattern.resolution = 480
        track = midi.Track()
        pattern.append(track)
        tickscale = 24
        lastcmdtime = 0
        prevstate = [0 for x in range(self.__pitch_span)]
        for time, state in enumerate(sequence):  
            offNotes = []
            onNotes = []
            for i in range(self.__pitch_span):
                n = state[i]
                p = prevstate[i]
                if p == 1 and n == 0:
                    offNotes.append(i)
                elif p == 0 and n == 1:
                    onNotes.append(i)
                pass
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, \
                                               pitch=note+self.__lowest_pitch))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, \
                                              velocity=80, pitch=note+self.__lowest_pitch))
                lastcmdtime = time
                pass
            prevstate = state
            pass
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
        midi.write_midifile(sp, pattern)
        pass
    
    def plot(self, fp, sp):
        sequence = self.parse(fp)
        plt.rcParams['figure.dpi'] = 300
        plt.imshow(sequence, aspect='auto')
        plt.savefig(sp)
        plt.rcParams['figure.dpi'] = 100
        plt.close()
        pass
    
    pass


class SequenceParser(object):
    '''Sequence Parser'''
    
    def __init__(self, name=None):
        # 64, 48, 32, 28, 24, 20, 16, 14, 12, 10, 8, 6, 4, 2 
        self.durations = [1, 3/4, 1/2, 7/16, 3/8, 5/16, 1/4, 7/32, 3/16, 5/32, 1/8, 3/32, 1/16, 1/32]
        self.name = name
        pass
    
    def get_notes(self, sequence):
        sequence = np.array(sequence)
        span = sequence.shape[-1]
        prevstate = [0 for x in range(span)]
        notes_active = []
        notes = []
        for time, state in enumerate(sequence):  
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                # note on
                if p == 0 and n == 1:
                    # pitch, start time, end time
                    note = [i, time, None]
                    notes_active.append(note)
                    pass
                # note off
                elif p == 1 and n == 0:
                    for note in notes_active:
                        if note[0] == i:
                            notes_active.remove(note)
                            note[2] = time
                            notes.append(note)
                            pass
                pass
            prevstate = state
            pass
        return notes
    
    def sort_notes(self, notes, mode=1, reverse=False):
        if mode not in (0, 1, 2):
            raise ValueError("mode must be in (1, 2, 3)")
        sorted_notes = []
        notes_copy = notes[:]
        while len(notes_copy) != 0:
            min_note = notes_copy[0]
            for note in notes_copy:
                if note[mode] < min_note[mode]:
                    min_note = note
                    pass
                pass
            sorted_notes.append(min_note)
            notes_copy.remove(min_note)
            pass
        if reverse:
            return sorted_notes[::-1]
        return sorted_notes
    
    def get_companion_matrix(self, sequence):
        sequence = np.array(sequence)
        span = sequence.shape[-1]
        notes = self.get_notes(sequence)
        notes = self.sort_notes(notes, mode=1, reverse=False)
        companion_matrix = np.zeros((span, span))
        while len(notes) > 1:
            note = notes[0]
            for n in notes[1:]:
                # note on at the same time
                if note[1] == n[1]:
                    companion_matrix[note[0]][n[0]] += 1
                    companion_matrix[n[0]][note[0]] += 1
                    pass
                elif note[1] < n[1]:
                    break
                pass
            notes.remove(note)
            pass
        return companion_matrix
    
    def get_transition_matrix(self, sequence):
        sequence = np.array(sequence)
        span = sequence.shape[-1]
        notes = self.get_notes(sequence)
        notes = self.sort_notes(notes, mode=1, reverse=False)
        transition_matrix = np.zeros((span, span))
        while len(notes) > 1:
            note = notes[0]
            last_start_time = note[1]
            for n in notes[1:]:
                if note[1] < n[1]:
                    # find the nearest notes
                    if last_start_time == note[1] or last_start_time == n[1]:
                        transition_matrix[note[0]][n[0]] += 1
                        last_start_time = n[1]
                        pass
                    else:
                        break
                    pass
                pass
            notes.remove(note)
            pass
        return transition_matrix
    
    def get_monosyllabic_melody(self, melody_sequence):
        sequence = np.array(melody_sequence)
        span = sequence.shape[-1]
        monosyllabic_melody = []
        for i in range(sequence.shape[0]):
            state = sequence[i]
            # rest
            if np.sum(state) == 0:
                melody_state = [0 for x in range(span)]
                monosyllabic_melody.append(melody_state)
                pass
            else:
                # select the highest pitch
                for pitch in range(span-1, -1, -1):                        
                    if state[pitch] == 1:
                        melody_state = [0 for x in range(span)]
                        melody_state[pitch] = 1
                        monosyllabic_melody.append(melody_state)              
                        break
                    pass
                pass
            pass
        monosyllabic_melody = np.array(monosyllabic_melody)
        return monosyllabic_melody
    
    def monosyllabic_melody_to_matrix(self, monosyllabic_melody):
        sequence = np.array(monosyllabic_melody)
        span = sequence.shape[1]
        durations = sorted(self.durations)
        # note + rest
        len_vector = (span + 1) * len(durations)
        notes = self.get_notes(sequence)
        notes = self.sort_notes(notes, mode=1, reverse=False)
        return_matrix = []
        durations_range = []
        cutting_lines = [1]
        for i in range(len(durations)-1):
            duration = int(durations[i] * 64)
            next_duration = int(durations[i + 1] * 64)
            cutting_line = (duration + next_duration) // 2
            cutting_lines.append(cutting_line)
            pass
        for i in range(len(cutting_lines)-1):
            durations_range.append((cutting_lines[i], cutting_lines[i+1]))
            pass
        assert(len(durations_range) == len(durations)-1)
        # note with longer duration will be ahead of schedule
        durations_range = sorted(durations_range, reverse=True)
        # to find rest
        last_end_time = notes[0][1]
        for note in notes:
            # rest
            if note[1] != last_end_time:
                vector = [0 for x in range(len_vector)]
                note_pitch = span
                note_duration = note[1] - last_end_time
                note_index = None
                for index, duration_range in enumerate(durations_range):
                    if note_duration in range(duration_range[0], duration_range[1]):  
                        note_index =  note_pitch + (span + 1) * (index + 1)
                        vector[note_index] = 1
                        return_matrix.append(vector)
                        break
                    pass
                if note_index == None:
                    note_index =  note_pitch
                    vector[note_index] = 1
                    return_matrix.append(vector)
                    pass
                pass
            # note
            vector = [0 for x in range(len_vector)]
            note_pitch = note[0]
            note_duration = note[2] - note[1]
            note_index = None
            for index, duration_range in enumerate(durations_range):
                if note_duration in range(duration_range[0], duration_range[1]):  
                    note_index =  note_pitch + (span + 1) * (index + 1)
                    vector[note_index] = 1
                    return_matrix.append(vector)
                    break
                pass
            if note_index == None:
                note_index =  note_pitch
                vector[note_index] = 1
                return_matrix.append(vector)
                pass
            # end time
            last_end_time = note[2]
            pass
        return_matrix = np.array(return_matrix)
        return return_matrix
    
    def matrix_to_monosyllabic_melody(self, matrix):
        matrix = np.array(matrix)
        len_vector = matrix.shape[1]
        span = len_vector // len(self.durations) - 1
        durations = sorted(self.durations, reverse=True)
        monosyllabic_melody = []
        for i in range(matrix.shape[0]):
            vector = matrix[i]
            note_index = np.argmax(vector)
            note_pitch = (note_index) % (span + 1)
            note_duration_index = note_index // (span + 1)
            note_duration = int(64 * durations[note_duration_index])
            if note_pitch != span:
                # note
                state = [0 for x in range(span)]
                state[note_pitch] = 1
                for j in range(note_duration):
                    monosyllabic_melody.append(state)
                    pass
                pass
            else:
                # rest
                state = [0 for x in range(span)]
                for j in range(note_duration):
                    monosyllabic_melody.append(state)
                    pass
                pass
            pass
        monosyllabic_melody = np.array(monosyllabic_melody)
        return monosyllabic_melody
    
    pass

