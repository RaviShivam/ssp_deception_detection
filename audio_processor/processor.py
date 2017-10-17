import subprocess
import os

data = "../data_audio/banana_dance.wav"
def extract_audio():
    for root, dirs, files in os.walk("../data_video/"):
        for f in files:
            command = "ffmpeg -i {}{} -ab 160k -ac 2 -ar 44100 -vn ../data_audio/{}.wav".format(root, f, f[:-4])
            subprocess.call(command, shell=True)


from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('../data_audio/banana_dance.wav')
times = np.arange(len(data))/float(samplerate)
timestep = times[1]

# data = data[:, 0]
silences = np.array([0 if (-1000<x<1000) else x for x in data[:, 0]])
print "time (seconds) in total: {}".format(len(times)*timestep)
print "time (seconds) with pauses: {}".format(sum(silences==0)*timestep)

plt.subplot(2,1,1)
plt.plot(times, data[:, 0])
plt.subplot(2,1,2)
plt.plot(times, silences)
plt.savefig('plot.png', dpi=1000)
plt.show()
