import matplotlib
gui_env = ['Qt5Agg','macosx','TKAgg','GTKAgg','Qt4Agg','WXAgg']
gui = gui_env[0]
matplotlib.use(gui,warn=False, force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import fftpack
from skimage import util
from SignalGenerator import signal_3



def plot_psd(x,Fs=100):
    plt.subplot(212)
    plt.psd(x,Fs=Fs)
    plt.show()

from fileReader import readFromAllFiles,readFromFileChFig1, readFromFileChFig3, readFromFileChFig4, readFromFileChFig5,readFromFileChFig6,readFromFileChFig7,readFromFileChFig8

realMeasurmentData = readFromFileChFig6()
realSignal = np.array(realMeasurmentData[1])
#realSignal, freq = signal_3()


freqs, times, Sx = signal.spectrogram(realSignal, fs=100, window='hanning',
                                      nperseg=128, noverlap=127,
                                      detrend=False, scaling='spectrum')
time = []
t = 0.00
dT = 0.01
for i in range(len(realSignal)):
    time.append(t)
    t += dT

f, ax = plt.subplots(figsize=(4.8, 2.4))
plt.ylim(0,20)
ax.pcolormesh(times, freqs , 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]')
#plt.plot(time,freq[0])
#plt.plot(time,freq[1])

np.savetxt('test.out', realSignal, delimiter=',')






plt.show()

