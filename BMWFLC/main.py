import matplotlib
gui_env = ['Qt5Agg','macosx','TKAgg','GTKAgg','Qt4Agg','WXAgg']
gui = gui_env[0]
matplotlib.use(gui,warn=False, force=True)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
#sns.set_palette(sns.color_palette("Blues"))
import sys
import math
from Filters import FLC, WFLC, BMFLC, EBMFLC, BMWFLC, BMWFLC
from SignalGenerator import complexSignal, signal_1, signal_freq, signal_amp, signal_multiple, signal_action_tremor
from fileReader import readFromAllFiles,readFromFileChFig1, readFromFileChFig3, readFromFileChFig4, readFromFileChFig5,readFromFileChFig6,readFromFileChFig7,readFromFileChFig8, readFromFileLog3, readFromTestUtenNr, readFromTestMasseNr, readFromTestDempingNr
import numpy as np
from scipy import signal
import re
from copy import deepcopy
import matplotlib.animation as animation
from collections import deque
from collections import OrderedDict
from matplotlib.legend_handler import HandlerLine2D
import csv
import pandas as pd
from sklearn import preprocessing

from colormaps import parula
from matplotlib.ticker import MaxNLocator



Writer = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = Writer(fps=60, metadata=metadata)


matplotlib.rcParams.update({'font.size': 60})


def plotPSD_DVA(x1, x2, x3, Fs=100, linear=True, save_to_file=False, rec_nr=0, sim_nr=0, DVA_nr=0, show=False, nperseg=1000):
    plt.clf()
    plt.hold(True)
    f1, Pxx_den_1 = signal.welch(x1, Fs, nperseg=nperseg, scaling='density')
    f2, Pxx_den_2 = signal.welch(x2, Fs, nperseg=nperseg, scaling='density')
    f3, Pxx_den_3 = signal.welch(x3, Fs, nperseg=nperseg, scaling='density')
    if linear:
        plt.plot(f1, Pxx_den_1)
        plt.plot(f2, Pxx_den_2)
        plt.plot(f3, Pxx_den_3)
    else:
        plt.semilogy(f1, Pxx_den_1)
        plt.semilogy(f2, Pxx_den_2)
        plt.semilogy(f3, Pxx_den_3)
        plt.gca().set_ylim(bottom=pow(10, -3))

    plt.xlim([0, 20])
    plt.xlabel('Frequency (Hz)', size=16)
    plt.ylabel(r'PSD ($(deg/sec)^2$/Hz)', size=16)
    plt.xticks(size=15)
    plt.yticks(size=15)

    peaksDict = {}
    magnitudePrev = 0
    magnitudeDiffPrev = 0
    peakFrequencies = []

    # Find peaks
    for i in range(len(Pxx_den_1)):
        magnitudeDiff = Pxx_den_1[i] - magnitudePrev
        if magnitudeDiff < 0 and magnitudeDiffPrev > 0:
            peaksDict[i - 1] = magnitudePrev
        # Check if last magnitude is a peak
        elif i == (len(Pxx_den_1) - 1) and magnitudeDiff > 0:
            peaksDict[i] = Pxx_den_1[i]

        magnitudeDiffPrev = magnitudeDiff
        magnitudePrev = Pxx_den_1[i]

    # Sort peaks by magnitude
    peaksDict = OrderedDict(sorted(peaksDict.items(), key=lambda t: t[1], reverse=True))
    allPeaksSorted = list(peaksDict.items())

    # plt.title(r"$f_{n_1}$: " + f"{f[allPeaksSorted[0][0]]:.4f} Hz           " + r"$f_{n_2}$: " + f"{f[allPeaksSorted[1][0]]:.4f} Hz")

    print(f"recording {rec_nr}")
    for i in range(10):
        print(f'Peak {i+1} Freq: {f1[allPeaksSorted[i][0]]:.4} Amp: {allPeaksSorted[i][1]:.4}')

    plt.tight_layout(pad=0.1)
    if save_to_file and rec_nr != 0:
        if linear:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/rec{rec_nr}Lin.pdf',
                format='pdf')
        else:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/rec{rec_nr}Log.pdf',
                format='pdf')
    elif save_to_file and sim_nr != 0:
        if linear:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/sim{sim_nr}Lin.pdf',
                format='pdf')
        else:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/sim{sim_nr}Log.pdf',
                format='pdf')
    elif save_to_file and DVA_nr != 0:
        if linear:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/DVA{DVA_nr}Lin.pdf',
                format='pdf')
        else:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/DVA{DVA_nr}Log.pdf',
                format='pdf')
    if show:
        plt.show()
    plt.clf()
    plt.close()

    for peak in allPeaksSorted:
        peakFrequencies.append((f1[peak[0]]))

    return peakFrequencies

def plotPSD(x, Fs=100, linear = True, save_to_file = False, rec_nr = 0, sim_nr=0,DVA_nr =0 ,show=False, nperseg=1000):
    f, Pxx_den = signal.welch(x, Fs ,nperseg=nperseg, scaling='density')
    if linear:
        plt.plot(f, Pxx_den)
    else:
        plt.semilogy(f, Pxx_den)
        plt.gca().set_ylim(bottom=pow(10, -3))

    plt.xlim([0, 20])
    plt.xlabel('Frequency (Hz)', size = 16)
    plt.ylabel(r'PSD ($(deg/sec)^2$/Hz)', size = 16)
    plt.xticks(size=15)
    plt.yticks(size=15)

    peaksDict = {}
    magnitudePrev = 0
    magnitudeDiffPrev = 0
    peakFrequencies = []

    # Find peaks
    for i in range(len(Pxx_den)):
        magnitudeDiff = Pxx_den[i] - magnitudePrev
        if magnitudeDiff < 0 and magnitudeDiffPrev > 0:
            peaksDict[i - 1] = magnitudePrev
        # Check if last magnitude is a peak
        elif i == (len(Pxx_den) - 1) and magnitudeDiff > 0:
            peaksDict[i] = Pxx_den[i]

        magnitudeDiffPrev = magnitudeDiff
        magnitudePrev = Pxx_den[i]

    # Sort peaks by magnitude
    peaksDict = OrderedDict(sorted(peaksDict.items(), key=lambda t: t[1], reverse=True))
    allPeaksSorted = list(peaksDict.items())

    #plt.title(r"$f_{n_1}$: " + f"{f[allPeaksSorted[0][0]]:.4f} Hz           " + r"$f_{n_2}$: " + f"{f[allPeaksSorted[1][0]]:.4f} Hz")

    print(f"recording {rec_nr}")
    for i in range(10):
        print(f'Peak {i+1} Freq: {f[allPeaksSorted[i][0]]:.4} Amp: {allPeaksSorted[i][1]:.4}')


    plt.tight_layout(pad=0.1)
    if save_to_file and rec_nr != 0:
        if linear:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/rec{rec_nr}Lin.pdf', format='pdf')
        else:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/rec{rec_nr}Log.pdf', format='pdf')
    elif save_to_file and sim_nr != 0:
        if linear:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/sim{sim_nr}Lin.pdf', format='pdf')
        else:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/sim{sim_nr}Log.pdf', format='pdf')
    elif save_to_file and DVA_nr != 0:
        if linear:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/DVA{DVA_nr}Lin.pdf', format='pdf')
        else:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/PSDplots/DVA{DVA_nr}Log.pdf', format='pdf')
    if show:
        plt.show()
    plt.clf()
    plt.close()

    for peak in allPeaksSorted:
        peakFrequencies.append((f[peak[0]]))

    return peakFrequencies

def plot_heatmap(data, save_to_file = False, rec_nr = 0, sim_nr = 0, show=False):

    df = pd.DataFrame(data=data[1:, 1:],
                 index=data[1:, 0],
                 columns=data[0, 1:])

    sns.set(style="white")

    # Generate a large random dataset

    # Set up the matplotlib figure
    #f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    #sns.heatmap(data, cmap=cmap, vmax=.3, center=0,
    #            square=True, linewidths=.5, cbar_kws={"shrink": .5})


    if rec_nr == 1:
        y_tick = 100
    elif rec_nr == 2:
        y_tick = 250
    elif rec_nr == 3:
        y_tick = 250
    elif rec_nr == 4:
        y_tick = 100
    elif rec_nr == 5:
        y_tick = 100
    elif rec_nr == 6:
        y_tick = 100
    elif rec_nr == 7:
        y_tick = 100
    elif rec_nr == 8:
        y_tick = 250
    elif rec_nr == 9:
        y_tick = 250
    elif rec_nr == 10:
        y_tick = 250
    elif rec_nr == 11:
        y_tick = 250
    elif rec_nr == 12:
        y_tick = 500
    elif rec_nr == 13:
        y_tick = 500
    elif rec_nr == 14:
        y_tick = 250
    elif rec_nr == 15:
        y_tick = 250
    elif sim_nr == 1:
        y_tick = 100
    elif sim_nr == 2:
        y_tick = 100
    elif sim_nr == 3:
        y_tick = 250
    elif sim_nr == 4:
        y_tick = 250
    elif sim_nr == 5:
        y_tick = 250
    elif sim_nr == 6:
        y_tick = 100
    elif sim_nr == 7:
        y_tick = 100
    elif sim_nr == 8:
        y_tick = 100
    elif sim_nr == 9:
        y_tick = 250
    elif sim_nr == 10:
        y_tick = 100



    ax = sns.heatmap(df, cmap=parula, yticklabels = 10, xticklabels = y_tick, linewidths = 0, rasterized=True, cbar_kws={'label': 'Magnitude'})
    ax.invert_yaxis()
    plt.yticks(rotation=0)

    #sns.set()
    plt.ylabel("Frequency (Hz)", size=16)
    plt.xlabel("Time (sec)", size=16)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout(pad=0.3)

    if save_to_file and rec_nr != 0:
        plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Heatmaps/heatmap{rec_nr}.pdf',format='pdf')
    elif save_to_file and sim_nr != 0:
        plt.savefig(
            f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Heatmaps/heatmap_sim{sim_nr}.pdf',
            format='pdf')
    if show:
        plt.show()
    plt.clf()
    plt.close()

    sns.set(style="whitegrid")



def combined_filters():
    filter2 = WFLC()
    time = 0.00
    realSignal = []
    estimatedSignal = []
    signalTime = []
    for i in range(10000):
        signalTime.append(time)
        SIG = simpleSignal(time, f1=f1, f2=f2)
        realSignal.append(SIG)
        estimatedSignal.append(filter3.BMFLC(time, SIG))
        time += timeStep
        print(filter3.estimatedFrequency)
    plt.plot(signalTime, realSignal)
    plt.plot(signalTime, estimatedSignal)
    plt.show()
    time = 0.00
    realSignal = []
    estimatedSignal = []
    estimatedSignal2 = []
    signalTime = []
    filter2.v0 = 2 * math.pi * filter3.estimatedFrequency
    for i in range(100000):
        signalTime.append(time)
        SIG = simpleSignal(time, f1=f1, f2=f3)
        realSignal.append(SIG)
        est = filter3.BMFLC(time, SIG)
        estimatedSignal.append(est)
        estimatedSignal2.append(filter2.WFLC(time, est))
        time += timeStep
    #
    plt.plot(signalTime, realSignal)
    plt.plot(signalTime, estimatedSignal)
    plt.plot(signalTime, estimatedSignal2)
    plt.show()
    time = 0.00
    realSignal = []
    estimatedSignal = []
    signalTime = []

def wflc_tuned():
    filter = WFLC(n=1,mu=0.01,mu0=0.000002 ,f0=4.5)
    f1=4
    timeStep = 0.01
    time = 0.00
    realSignal = []
    estimatedSignal = []
    realFreq = []
    estimatedFreq = []
    signalTime = []

    plt.ion()
    for i in range(10000):
        signalTime.append(time)
        a, b, n = simpleSignal(time, f1=f1, f2=f2)
        SIG = a
        realSignal.append(SIG)
        est = filter.WFLC(time, SIG)
        estimatedSignal.append(est)
        estimatedFreq.append(filter.estimatedFrequency)
        realFreq.append(f1)

        if i > 200 and i <= 300:
            f1 += 0.01

        if i > 700 and i <= 800:
            f1 -= 0.005



        time += timeStep
        #plt.ylim(0,20)
        #plt.plot(filter.estimatedFrequency, math.sqrt(filter.W[0][0] ** 2 + filter.W[1][0] ** 2), "ob'")
        #plt.draw()
        #plt.pause(0.00001)
        #plt.clf()

    plt.ioff()

    plt.plot(signalTime, realSignal)
    plt.plot(signalTime, estimatedSignal)
    plt.show()
    plt.clf()
    plt.ylim(3.9,7)
    plt.plot(signalTime, realFreq, "--")
    plt.plot(signalTime, estimatedFreq)
    plt.show()

def run_BMWFLC(f_min = 4, f_max = 6, dT = 0.01, dF = 0.1, plot_update_rate = 100, plot = False, peaks_to_track = 2,
               tremor_data = readFromFileChFig1(), mu=0, kappa = 0.009, g = 100, h = 0.00001, Tp=2.0, alpha=0.67,
               beta=100, l=0.1, frequencies = [], use_stem=True, save_to_file= False, rec_nr = 0, y_min = 3,y_max=20,
               show = True, adaptive_lr=True, sim_nr = 0, error_ymax = 2, error_ymax_zoom = 0.065, plot_real_frequency=True):
    
    _filter = BMWFLC(mu=mu, f_min=f_min, f_max=f_max, dF=dF, dT=dT, peaks_to_track = peaks_to_track, kappa = kappa, g = g, h = h, Tp=Tp, alpha=alpha, beta=beta, l=l, adaptive_lr=adaptive_lr)
    print(len(_filter.V))

    data_points = len(tremor_data)

    if len(frequencies) == 0:
        frequencies = np.zeros(shape=(2,data_points))

    time_elapsed = 0.0
    estimated_signal = []
    estimated_frequency = np.zeros(shape=(_filter.peaks_to_track, len(tremor_data)))
    error = np.zeros(shape=(_filter.peaks_to_track, len(tremor_data)))
    signalTime = []
    noiseSignal = []
    band_frequencies = []
    frequencies_real = []
    #frequencies_real.append(frequencies[0][0])
    #frequencies_real.append(frequencies[1][0])

    # Heatmap stuff
    heatmapd_updaterate = 1
    heatmap_header = [0]
    heatmap_time = 0
    heatmap_data= []

    for index in range(_filter.n) :
        heatmap_data.append(np.array([np.array(_filter.Vref).T[index] / (2 * math.pi)]))

    heatmap_data = np.array(heatmap_data)

    for index  in range(len(tremor_data)):
        if index % heatmapd_updaterate == 0:
            heatmap_header.append(heatmap_time)
        heatmap_time += dT
    heatmap_header.append(heatmap_time)

    heatmap_magnitudes = []
    for index in range(_filter.n):
        heatmap_magnitudes.append(np.array([_filter.magnitudes[index]]))
    heatmap_magnitudes = np.array(heatmap_magnitudes)
    heatmap_data = np.concatenate((heatmap_data, heatmap_magnitudes), axis=1)
    # Heatmap stuff

    # Turn interactive mode on
    plt.ion()

    for i in range(len(tremor_data)):
        signalTime.append(time_elapsed)

        SIG = tremor_data[i]

        est = _filter.BMWFLC(time_elapsed, SIG)

        estimated_signal.append((est))
        time_elapsed += dT


        for peak in _filter.allPeaksSorted[:2]:
            plt.plot(band_frequencies[peak[0]], _filter.magnitudes[peak[0]], "bo")

        if peaks_to_track > 0:
            for p in range(peaks_to_track):
                if len(_filter.allPeaksSorted) > p:
                    estimated_frequency[p][i] = _filter.V[_filter.allPeaksSorted[p][0]] / (2 * math.pi)
                    if sim_nr != 0:
                        error[p][i] =  abs(frequencies[p][i] - (_filter.V[_filter.allPeaksSorted[p][0]] / (2 * math.pi)))



        # Heatmap stuff

        if i % heatmapd_updaterate == 0:
            heatmap_magnitudes = []
            for index in range(_filter.n):
                heatmap_magnitudes.append(np.array([_filter.magnitudes[index]]))
            heatmap_magnitudes = np.array(heatmap_magnitudes)
            heatmap_data = np.concatenate((heatmap_data, heatmap_magnitudes), axis=1)

        # Heatmaps stuff

        if i %  plot_update_rate == 0 or i == (len(tremor_data)-1):

            if i == (len(tremor_data)-1):
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
                plt.xlabel("Frequency, Hz")
            else:
                plt.plot(band_frequencies, _filter.magnitudes)
                plt.plot(band_frequencies, _filter.magnitudes, "ro")
            for peak in _filter.allPeaksSorted[:_filter.peaks_to_track]:
                plt.plot(band_frequencies[peak[0]], _filter.magnitudes[peak[0]], "ro", zorder=2)
            peak1 = 0
            peak2 = 0
            if len(_filter.allPeaksSorted) > 0:
                peak1 =_filter.V[_filter.allPeaksSorted[0][0]]
            if len(_filter.allPeaksSorted) > 1:
                peak2 =_filter.V[_filter.allPeaksSorted[1][0]]

            #plt.title(f"f1: {frequencies[0][i]:.4f} f2: {frequencies[1][i]:.4f} Time:{time_elapsed:.1f} Estimate1: {peak1/ (2 * math.pi):.4f} Estimate2: {peak2/(2*math.pi):.4f}")
            #plt.title(f"Time:{time_elapsed:.1f} Estimate1: {peak1/ (2 * math.pi):.4f} Estimate2: {peak2/(2*math.pi):.4f}")
            label = 'Line 1'

            if i == (len(tremor_data)-1):
                heatmap_data = np.insert(heatmap_data, 0, heatmap_header, axis = 0)
                plt.tight_layout(pad=0.1)
                if show:
                    plt.show()
            else:
                if show:
                    plt.draw()
                    plt.pause(0.00001)

    print(f"Time elapsed: {time_elapsed:.2f} Seconds")
    print()

    if plot:
        plt.clf()
        plt.close()
        plot_heatmap(heatmap_data, save_to_file, rec_nr, sim_nr ,show=show)
        if show:
            plt.show()
        plt.clf()
        plt.close()
        _ = plotPSD(tremor_data, linear=False, save_to_file=save_to_file, rec_nr=rec_nr, show=show, sim_nr=sim_nr)
        PSDestimate = plotPSD(tremor_data,  save_to_file=save_to_file, rec_nr=rec_nr, show=show, sim_nr=sim_nr)
        plt.clf()
        plt.close()
        # Plot signal
        plt.figure(figsize=(12, 3))
        plt.ylabel("(deg/sec)", size = 16)
        plt.xlabel("Time (sec)", size = 16)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.plot(signalTime, tremor_data)
        #plt.plot(signalTime, estimated_signal)
        np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/tremor_signal_{rec_nr}.csv', tremor_data, delimiter=',')
        np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/estimated_frequency_{rec_nr}.csv', estimated_frequency, delimiter=',')
        #plt.plot(signalTime, noiseSignal)
        plt.tight_layout(pad=0.1)
        if save_to_file and rec_nr != 0:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Signals/signal{rec_nr}.pdf',format='pdf')
        elif save_to_file and sim_nr != 0:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Signals/signal_sim{sim_nr}.pdf',
                format='pdf')
        if show:
            plt.show()
        plt.clf()
        plt.close()

        plt.ylim([f_min,f_max])
        #plt.plot(signalTime, dominant_frequencies, '-k')
        # Plot peak frequencies from PSD
        plt.figure(figsize=(12, 5))
        plt.ylabel("Frequency (Hz)", size=16)
        plt.xlabel("Time (sec)", size=16)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.ylim([y_min, y_max])
        #a = np.empty(len(signalTime))
        #b = np.empty(len(signalTime))
        #c = np.empty(len(signalTime))
        #d = np.empty(len(signalTime))

        #a.fill(PSDestimate[0])
        #b.fill(PSDestimate[1])
        #c.fill(PSDestimate[2])
        #d.fill(PSDestimate[3])

        if plot_real_frequency:

            if(frequencies[0][0] > frequencies[1][0]):
                a = frequencies[0]
                b = frequencies[1]
            else:
                b = frequencies[0]
                a = frequencies[1]

            plt.plot(signalTime, b, '--k', label=r'$f_1$')
            plt.plot(signalTime, a, '-k', label=r'$f_2$')
            #plt.plot(signalTime, c, '-.k', label=r"Peak 3")
            #plt.plot(signalTime, d, ':k', label=r"Peak 4")

        #Reset color cycle
        plt.gca().set_prop_cycle(None)

        # Plot estimated frequencies

        for i in range(len(estimated_frequency)):
            line, = plt.plot(signalTime, estimated_frequency[i], ".", label=f"Estimate {i+1}")

        plt.legend(frameon=True, prop={'size': 15}, markerscale=3)
        plt.tight_layout(pad=0.1)
        if save_to_file and rec_nr != 0:
            plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Estimates/estimate{rec_nr}.pdf',format='pdf')
        elif save_to_file and sim_nr != 0:
            plt.savefig(
                f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Estimates/estimate_sim{sim_nr}.pdf',
                format='pdf')
        if show:
            plt.show()
        plt.clf()
        plt.close()


        #Plot error
        if sim_nr != 0:
            plt.figure(figsize=(12, 4))
            plt.ylabel("Error (Hz)", size=16)
            plt.xlabel("Time (sec)", size=16)
            plt.xticks(size=15)
            plt.yticks(size=15)
            plt.ylim([0, error_ymax])
            for i in range(len(error)):
                line, = plt.plot(signalTime, error[i], label=f"Estimate {i+1} error")
            plt.legend(frameon=True, prop={'size': 15}, markerscale=3)
            plt.tight_layout(pad=0.1)
            if save_to_file:
                plt.savefig(
                    f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/ErrorPlots/ErrorPlot_sim{sim_nr}.pdf',
                    format='pdf')
            if show:
                plt.show()

        # Plot error Zoom
        if sim_nr != 0:
            plt.figure(figsize=(12, 3))
            plt.ylabel("Error (Hz)", size=16)
            plt.xlabel("Time (sec)", size=16)
            plt.xticks(size=15)
            plt.yticks(size=15)
            plt.ylim([0, error_ymax_zoom])
            for i in range(len(error)):
                line, = plt.plot(signalTime, error[i], label=f"Estimate {i+1} error")
            plt.legend(frameon=True, prop={'size': 15}, markerscale=3)
            plt.tight_layout(pad=0.1)
            if save_to_file:
                plt.savefig(
                    f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/ErrorPlots/ErrorPlot_sim{sim_nr}_Zoom.pdf',
                    format='pdf')
            if show:
                plt.show()
        plt.clf()
        plt.close()


def plot_rec_1():
    simulated_signal, frequencies = complexSignal(4.6, 5.2, 128, 153, data_points=1000)

    run_BMWFLC(plot=True,frequencies = frequencies ,tremor_data=readFromFileChFig1()[0],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2, error_ymax=1.25, error_ymax_zoom=0.11,
               plot_update_rate=1000, save_to_file = False,sim_nr=99 ,rec_nr=1, y_min=3, y_max=7, show=True)

def plot_rec_2():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig3()[0],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.1, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=2, y_min=3, y_max=7, show=False)

def plot_rec_3():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig3()[1],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.1, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=3, y_min=3, y_max=7, show=False)

def plot_rec_4():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig4()[0],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.1, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=4, y_min=3, y_max=7, show=False)

def plot_rec_5():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig4()[1],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.1, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=5, y_min=3, y_max=7, show=False)

def plot_rec_6():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig5()[0],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.1, peaks_to_track=4,
               plot_update_rate=1000, save_to_file = True, rec_nr=6, y_min=3, y_max=12, show=True)

def plot_rec_7():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig5()[1], mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.1, peaks_to_track=6,
               plot_update_rate=1000, save_to_file=False, rec_nr=7, y_min=4, y_max=11, show=True)

def plot_rec_8():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig6()[0],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=8, y_min=3, y_max=7, show=False)

def plot_rec_9():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig6()[1],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=9, y_min=3, y_max=7, show=False)

def plot_rec_10():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig6()[2],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=10, y_min=3, y_max=7, show=False)

def plot_rec_11():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig6()[3],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=11, y_min=3, y_max=7, show=False)

def plot_rec_12():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig7()[0],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=12, y_min=3, y_max=7, show=False)

def plot_rec_13():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig7()[1],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=13, y_min=3, y_max=7, show=False)


def plot_rec_14():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig8()[0],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file = True, rec_nr=14, y_min=3, y_max=7, show=False)

def plot_rec_15():
    run_BMWFLC(plot=True, tremor_data=readFromFileChFig8()[1],mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1,dT=0.01, Tp=2, alpha=0.67, beta=50,l=0.2, peaks_to_track=4,
               plot_update_rate=1000, save_to_file = False, rec_nr=15, y_min=3, y_max=7, show=True)

def plot_sim_1_example():
    simulated_signal, frequencies = complexSignal(4, 5, 4, 2, data_points = 1000, noisy=False)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=7, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=6000, save_to_file=False, rec_nr=0, sim_nr=1,y_min=4, y_max=6, show=True)

# Sim of recording 1 normal amplitude, wih adaptive learningrate and decay
def plot_sim_1():
    simulated_signal, frequencies = complexSignal(4.65, 5.25, 128, 153, data_points = 6000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=6000, save_to_file=False, rec_nr=0, sim_nr=1,y_min=4, y_max=6, show=True,error_ymax=1.25, error_ymax_zoom=0.065)

# Sim of recording 1 with voluntary motion
def plot_sim_2():
    simulated_signal, frequencies = signal_action_tremor(4.65, 5.25, 0.6, 128, 153, 500, data_points = 6000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=2,y_min=4, y_max=6, show=True, error_ymax=1.25, error_ymax_zoom=0.065)

# Varying amplitude
def plot_sim_3():
    simulated_signal, frequencies = signal_amp(f1=4.65, f2=5.25, a1=300, a2=400, data_points = 3000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=3,y_min=4, y_max=6, show=True, error_ymax=1.25, error_ymax_zoom=0.065)

# Varying frequency
def plot_sim_4():
    simulated_signal, frequencies = signal_freq(f1=4.05, f2=5.05, a1=300, a2=400, data_points = 3000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=4, y_min=3, y_max=6, show=True, adaptive_lr=True, error_ymax=2, error_ymax_zoom=0.065)

# 10 frequencies
def plot_sim_5():
    simulated_signal, frequencies = signal_multiple(f1=4.05, f2=5.05, a1=300, a2=400, data_points = 1000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=6,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=5, y_min=3, y_max=20, show=False, adaptive_lr=True, error_ymax=1.25, error_ymax_zoom=0.065, plot_real_frequency=False)


# h = 0. Not BMWFLC anymore, only BMFLC
def plot_sim_6():
    simulated_signal, frequencies = complexSignal(4.65, 5.25, 128, 153, data_points = 1000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=6,y_min=4, y_max=6, show=False)

# Adaptive learning rate off, and decaying learningrate off. mu0 and mu constant
def plot_sim_7():
    simulated_signal, frequencies = complexSignal(4.65, 5.25, 128, 153, data_points = 1000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0.001, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=7,y_min=4, y_max=6, show=False, adaptive_lr=False)

# Decaying learningrate off, beta = 0
def plot_sim_8():
    simulated_signal, frequencies = complexSignal(4.65, 5.25, 128, 153, data_points = 1000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0.001, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=0, l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=8,y_min=4, y_max=6, show=False)

# Adaptive learning rate off, and decaying learningrate off. mu0 and mu constant Varying amplitude
def plot_sim_9():
    simulated_signal, frequencies = signal_amp(4.65, 5.25, 128, 153, data_points = 3000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0.001, kappa=0.01, g=200, h=0.00001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=1000, save_to_file=False, rec_nr=0, sim_nr=9,y_min=4, y_max=6, show=False, adaptive_lr=False, error_ymax=1.75, error_ymax_zoom=0.75,)

# Sim of recording 1 normal amplitude, wih adaptive learningrate and decay nice frequency
def plot_sim_10():
    simulated_signal, frequencies = complexSignal(4.6, 5.2, 128, 153, data_points = 1000)
    run_BMWFLC(plot=True, frequencies=frequencies, tremor_data=simulated_signal, mu=0, kappa=0.01, g=200, h=0.0001,
               f_min=3, f_max=20, dF=0.1, dT=0.01, Tp=2, alpha=0.67, beta=50, l=0.2, peaks_to_track=2,
               plot_update_rate=6000, save_to_file=False, rec_nr=0, sim_nr=10,y_min=4, y_max=6, show=True,error_ymax=1.25, error_ymax_zoom=0.006)

def plot_test(nr, data, dT=0.01):
    time=[]
    time_elapsed = 0

    for i in range(len(data[0])):
        time.append(time_elapsed)
        time_elapsed += dT

    # Plot signal
    plt.figure(figsize=(12, 3))
    plt.ylabel("(deg/sec)", size=16)
    plt.xlabel("Time (sec)", size=16)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.plot(time, data[0])
    plt.tight_layout(pad=0.1)
    plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Signals/signal_DVA{nr}.pdf',format='pdf')

    #plt.title(f"Test {nr} Gyro")

    plt.show()
    plt.clf()
    plt.close()
    #plt.title(f"Test {nr} Accel")
    #plt.plot(time, data[1])
    #plt.show()
    #plt.clf()
    #plt.close()
    #plt.title(f"Test {nr} PSD linear")
    plotPSD(data[0], Fs=100, linear=True, show=True, DVA_nr=nr, save_to_file=True, nperseg=1024)
    plotPSD(data[0], Fs=100, linear=False, show=True, DVA_nr=nr, save_to_file=True, nperseg=1024)
    plt.clf()
    plt.close()

    plt.ylim([3, 6])
    plt.plot(time, data[2], ".", label=f"Estimate {1}")
    plt.legend(frameon=True, prop={'size': 15}, markerscale=3)
    plt.tight_layout(pad=0.1)
    plt.savefig(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/LaTeX/NTNU Project Thesis/fig/Estimates/estimate_DVA{nr}.pdf',format='pdf')
    #plt.plot(time, data[3], ".", label=f"Estimate {2}")
    plt.show()
    plt.clf()
    plt.close()

while True:
    # Close frequencies
    #tremor_signal, frequencies = complexSignal(f1=4.6, f2=5.2, a1=300, a2=400, data_points = 6000)


    #run_BMWFLC(f_min=4, f_max=8,plot=True,beta=100, tremor_data=tremor_signal, frequencies = frequencies, use_stem=True)

    #signal, frequencies = signal_1(f1=5, f2=6.2, a1=300, a2=400)
    #run_BMWFLC(f_min=4, f_max=8,plot=True,beta=1000, tremor_data=signal, frequencies = frequencies, use_stem=True)

    #signal, frequencies = signal_2(f1=5, f2=7, a1=400, a2=300, data_points = 3000)
    #run_BMWFLC(f_min=4, f_max=8,plot=True,beta=1000, tremor_data=signal, frequencies = frequencies, use_stem=True)
    #run_BMWFLC(plot=True, tremor_data=readFromFileChFig7()[0],mu=1, d=0.009, g=100, h=0.001, f_min=4, f_max=8, dF=0.1, dT=0.01, Tp=1, alpha=0.67, beta=100,l=0.1, peaks_to_track=1)
    #run_BMWFLC(plot=True, tremor_data=signal,frequencies = frequencies,mu=1, d=0.009, g=100, h=0.001, f_min=4, f_max=8, dF=0.1, dT=0.01, Tp=1, alpha=0.67, beta=100,l=0.1, peaks_to_track=1)

    #un_BMWFLC(plot=True, tremor_data=readFromFileChFig1()[0],mu   =0, kappa = 0.01, g = 200, h = 0.0001, f_min = 3, f_max = 20, dF = 0.1, dT = 0.01, Tp = 2, alpha = 0.67, beta = 10, l = 0.1, peaks_to_track = 50, plot_update_rate = 1000, save_to_file = False, rec_nr = 1)

    #plot_rec_1()
    #plot_rec_2()
    #plot_rec_3()
    #plot_rec_4()
    #plot_rec_5()
    #plot_rec_6()
    #plot_rec_7()
    #plot_rec_8()
    #plot_rec_9()
    #plot_rec_10()
    #plot_rec_11()
    #plot_rec_12()
    #plot_rec_13()
    #plot_rec_14()
    plot_rec_15()

    #plot_sim_1_example()

    plot_sim_1()
    #plot_sim_2()
    #plot_sim_3()
    #plot_sim_4()
    #plot_sim_5()
    #plot_sim_6()
    #plot_sim_7()
    #plot_sim_8()
    #plot_sim_9()
    #plot_sim_10()


    # ----------------- DVA signals ---------------------
    nr = 2
    #plot_test(nr, readFromTestUtenNr(nr))
    #plot_test(nr, readFromTestMasseNr(nr))
    #plot_test(nr, readFromTestDempingNr(nr))

    #np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/tremor_signal_DVA_without{nr}.csv', readFromTestUtenNr(nr)[0],delimiter=',')
    #np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/estimated_frequency_DVA_without{nr}.csv',[readFromTestUtenNr(nr)[2], readFromTestDempingNr(nr)[3]], delimiter=',')

    #np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/tremor_signal_DVA_mass{nr}.csv',readFromTestMasseNr(nr)[0], delimiter=',')
    #np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/estimated_frequency_DVA_mass{nr}.csv',[readFromTestMasseNr(nr)[2], readFromTestDempingNr(nr)[3]], delimiter=',')

    #np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/tremor_signal_DVA_damp{nr}.csv',readFromTestDempingNr(nr)[0], delimiter=',')
    #np.savetxt(f'/Users/bjartesunde/Dropbox/NTNU/Masteroppgave/MatLab/estimated_frequency_DVA_damp{nr}.csv',[readFromTestDempingNr(nr)[2], readFromTestDempingNr(nr)[3]], delimiter=',')

    # ----------------- 3 in 1 ----------------
    #plotPSD_DVA(readFromTestUtenNr(nr)[0], readFromTestMasseNr(nr)[0], readFromTestDempingNr(nr)[0], Fs=100, linear=True, show=True, DVA_nr=nr, save_to_file=True, nperseg=1000)
    #plt.clf()
    # ----------------- 3 in 1 ----------------
    # ----------------- DVA signals ---------------------



    sys.exit()