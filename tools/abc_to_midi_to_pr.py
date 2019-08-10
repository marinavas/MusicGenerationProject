import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
import pretty_midi

import imageio


i = 0


data_x = []
data_prev = []

for file in os.listdir('data\Data_abc\Midi'):
    if file.endswith(".mid"):
        path =os.path.join('data\Data_abc\Midi', file)
        #print(path)
        

        midi_pretty_format = pretty_midi.PrettyMIDI(path)
        piano_midi = midi_pretty_format.instruments[0] # Get the piano channels
        piano_roll = piano_midi.get_piano_roll(fs = 4)

        #print(np.max(piano_roll))


        piano = np.zeros((128,piano_roll.shape[1]+16))
        piano[:,16:] = piano_roll

        #matplotlib.pyplot.imsave("data/data128/"+str(i)+".png",piano,cmap = "gray")
        i+=1
        
        if(i%10 == 0):
                print(i)

        for w in range(0,piano_roll.shape[1]-16,16): 
                #print(piano.shape)
                #print((piano[:,w+16:w+32].T).shape)
                data_x.append(piano[:,w+16:w+32].T)
                data_prev.append(piano[:,w:w+16].T)

data_x =np.array(np.reshape(data_x,(-1,16,128,1)))
data_prev =np.array(np.reshape(data_prev,(-1,16,128,1)))
np.save("data/data_x.npy",data_x)
np.save("data/data_prev.npy",data_prev)

print(data_x.shape)
        

