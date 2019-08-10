import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
import pretty_midi

import imageio


i = 0

for file in os.listdir('./Data/Midi2'):
    if file.endswith(".mid"):
        path =os.path.join('./Data/Midi2', file)
        print(path)
        

        midi_pretty_format = pretty_midi.PrettyMIDI(path)
        piano_midi = midi_pretty_format.instruments[0] # Get the piano channels
        piano_roll = piano_midi.get_piano_roll(fs = 4)

        print(np.max(piano_roll))

        piano = np.array([piano_roll,piano_roll,piano_roll])
        for w in range(0,piano_roll.shape[1]-64,64):
                
                matplotlib.pyplot.imsave("Data64/"+str(i)+".png",piano_roll[50:114,w:w+64],cmap = "gray")
                i+=1



