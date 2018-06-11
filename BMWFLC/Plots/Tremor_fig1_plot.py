import matplotlib
from scipy import signal
gui_env = ['Qt5Agg', 'macosx', 'TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
gui = gui_env[0]
matplotlib.use(gui, warn=False, force=True)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import sys
import math
from Filters import FLC, WFLC, BMFLC, EBMFLC, BMWFLC, BMWFLC
from SignalGenerator import complexSignal, signal_1, signal_2
from fileReader import readFromAllFiles, readFromFileChFig1, readFromFileChFig3, readFromFileChFig4, readFromFileChFig5, \
    readFromFileChFig6, readFromFileChFig7, readFromFileChFig8
import numpy as np
import matplotlib.animation as animation


Writer = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = Writer(fps=60, metadata=metadata)

matplotlib.rcParams.update({'font.size': 60})

def plotPSD(x, Fs=100):
    f, Pxx_den = signal.welch(x, Fs ,nperseg=2048, scaling='spectrum')
    plt.semilogy(f, Pxx_den)
    plt.xlim([0, 20])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    #f, Pxx_spec = signal.welch(x, Fs,nperseg=2048, scaling='spectrum')
    #plt.figure()
    #plt.semilogy(f, np.sqrt(Pxx_spec))
    #plt.xlim([0, 20])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('Linear spectrum [V RMS]')
    plt.show()


def run_BMWFLC(f_min=4, f_max=6, dT=0.01, dF=0.1, plot_update_rate=100, plot=False, peaks_to_track=2,
               tremor_data=readFromFileChFig1(), mu=1, kappa=0.009, g=100, h=0.00001, Tp=2.0, alpha=0.67, beta=100, l=0.1,
               frequencies=[], use_stem=True):
    _filter = BMWFLC(mu=0, f_min=f_min, f_max=f_max, dF=dF, dT=dT, peaks_to_track=peaks_to_track, kappa=kappa, g=g, h=h, Tp=Tp,
                     alpha=alpha, beta=beta, l=l)
    print(len(_filter.V))

    data_points = len(tremor_data)

    if len(frequencies) == 0:
        frequencies = np.zeros(shape=(2, data_points))

    time_elapsed = 0.0
    estimated_signal = []
    estimated_frequency = []
    signalTime = []
    noiseSignal = []
    dominant_frequencies = []
    band_frequencies = []

    # Turn interactive mode on
    plt.ion()

    for i in range(len(tremor_data)):
        signalTime.append(time_elapsed)

        SIG = tremor_data[i]

        est = _filter.BMWFLC(time_elapsed, SIG)

        estimated_signal.append((est))
        time_elapsed += dT

        if len(frequencies) > 0:
            dominant_frequencies.append((frequencies[0][i], frequencies[1][i]))

        for peak in _filter.allPeaksSorted[:2]:
            plt.plot(band_frequencies[peak[0]], _filter.magnitudes[peak[0]], "bo")

        if len(_filter.allPeaksSorted) > 1 and _filter.peaks_to_track >= 2:
            estimated_frequency.append((_filter.V[_filter.allPeaksSorted[0][0]] / (2 * math.pi),
                                        _filter.V[_filter.allPeaksSorted[1][0]] / (2 * math.pi)))
        elif len(_filter.allPeaksSorted) > 0:
            estimated_frequency.append((_filter.V[_filter.allPeaksSorted[0][0]] / (2 * math.pi), 0))
        else:
            estimated_frequency.append((0, 0))

        if i % plot_update_rate == 0 or i == (len(tremor_data) - 1):

            if i == (len(tremor_data) - 1):
                plt.ioff()
                plt.clf()

            band_frequencies = []

            for j in range(_filter.n):
                band_frequencies.append(_filter.V[j] / (2 * math.pi))
            plt.clf()
            plt.ylim([0, 3])
            if use_stem:
                (markerline, stemlines, baseline) = plt.stem(band_frequencies, _filter.magnitudes)
                plt.setp(baseline, visible=False)
                plt.setp(markerline, zorder=1)
                plt.ylabel("Magnitude")
                plt.xlabel("Frequency (Hz)")
            else:
                plt.plot(band_frequencies, _filter.magnitudes)
                plt.plot(band_frequencies, _filter.magnitudes, "ro")
            for peak in _filter.allPeaksSorted[:_filter.peaks_to_track]:
                plt.plot(band_frequencies[peak[0]], _filter.magnitudes[peak[0]], "ro", zorder=2)
            peak1 = 0
            peak2 = 0
            if len(_filter.allPeaksSorted) > 0:
                peak1 = _filter.V[_filter.allPeaksSorted[0][0]]
            if len(_filter.allPeaksSorted) > 1:
                peak2 = _filter.V[_filter.allPeaksSorted[1][0]]

            plt.title(f"f1: {frequencies[0][i]:.4f} f2: {frequencies[1][i]:.4f} Time:{time_elapsed:.1f} Estimate1: {peak1/ (2 * math.pi):.4f} Estimate2: {peak2/(2*math.pi):.4f}")

            if i == (len(tremor_data) - 1):
                plt.show()
            else:
                plt.draw()
                plt.pause(0.00001)

    print(f"Time elapsed: {time_elapsed:.2f} Seconds")
    print()

    if plot:
        plt.clf()
        plotPSD(tremor_data)
        plt.clf()
        plt.close()
        # Plot signal
        plt.figure(figsize=(12, 3))
        plt.ylabel("(deg/sec)", size=16)
        plt.xlabel("Time (sec)", size=16)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.tight_layout(pad=0.2)
        plt.plot(signalTime, tremor_data)
        #plt.plot(signalTime, estimated_signal)
        # plt.plot(signalTime, noiseSignal)
        plt.show()
        plt.clf()
        plt.ylim([f_min, f_max])
        x, y = zip(*estimated_frequency)

        plt.plot(signalTime, x, ".")
        plt.plot(signalTime, y, ".")
        plt.plot(signalTime, dominant_frequencies, "--")
        plt.show()


while True:
    # Magnitude spectrum
    real_signal = readFromFileChFig1()[0]
    run_BMWFLC(f_min=3, f_max=7, plot=True, beta=100, tremor_data=real_signal, use_stem=True, plot_update_rate=1000)
    #simulated_signal, frequencies = complexSignal(4.6,5.2,150,200)
    #run_BMWFLC(f_min=3, f_max=7, plot=True, beta=100, tremor_data=simulated_signal, use_stem=True, plot_update_rate=1000)




    sys.exit()