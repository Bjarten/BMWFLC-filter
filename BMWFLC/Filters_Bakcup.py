import numpy as np
import math
from copy import deepcopy

class FLC():

    """ FLC filter class

    Attributes
    ----------
    n : int
        Number of harmonics
    X : ndarray
        Reference input vector
    W : ndarray
        Weights
    V : ndarray
        Angular  frequencies
    mu : float
        Adaptive filter gain
    f0 : float
        frequency
    """

    def __init__(self, n=5, mu=0.07, f0=8):
        """
        Parameters
        ----------
        n : int
            Number of harmonics
        mu : float
            Adaptive filter gain
        f0 : float
            Frequency of input signal
        """

        self.n = n
        self.mu = mu
        self.f0 = f0
        self.X = np.zeros(shape=(2, n))
        self.W = np.zeros(shape=(2, n))
        self.V = np.array(np.zeros([n]))

        for i in range(self.n):
            self.V[i] = i * 2 * math.pi * self.f0;


    def FLC(self, k, s):
        """ FLC filter

        Parameters
        ----------
        k : float
            Time instant
        s : float
            Reference signal

        Returns
        -------
        y : float
            Estimated signal
        """

        # Find reference input vector
        for i in range(self.n):
            self.X[0][i] = math.sin(self.V[i] * k)
            self.X[1][i] = math.cos(self.V[i] * k)
        # Find estimated signal
        y = np.dot(np.transpose(self.W[0]), self.X[0]) + np.dot(np.transpose(self.W[1]), self.X[1])

        err = s - y
        # Update weights

        self.W[0] += 2 * self.mu * self.X[0] * err
        self.W[1] += 2 * self.mu * self.X[1] * err

        return y


class WFLC():
    """ WFLC filter class

    Attributes
    ----------
    n : int
        Number of harmonics
    X : ndarray
        Reference input vector
    W : ndarray
        Weights
    V : ndarray
        Angular  frequencies
    mu : float
        Adaptive filter gain
    v0 : float
        ω0 fundamental angular frequency
    """

    def __init__(self, n=2, mu=0.001, mu0=0.000001, f0 = 6):
        """
        Parameters
        ----------
        n : int
            Number of harmonics
        mu : float
            Adaptive filter gain for reference input vector
        mu0 : float
            Adaptive filter gain for fundamental frequency
        """

        self.n = n
        self.mu = mu
        self.mu0 = mu0
        self.v0 = 2*math.pi*f0
        self.X = np.zeros(shape=(2, n))
        self.W = np.zeros(shape=(2, n))
        self.V = np.array(np.zeros([n]))
        self.estimatedFrequency = 0


    def WFLC(self,k, s):
        """ FLC filter

        Parameters
        ----------
        k : float
            Time instant
        s : float
            Reference signal

        Returns
        -------
        y : float
            Estimated signal
        """


        # Find reference input vector
        for i in range(self.n):
            self.X[0][i] = math.sin((i+1) * self.v0 * k)
            self.X[1][i] = math.cos((i+1) * self.v0 * k)
        # Find estimated signal
        y = np.dot(np.transpose(self.W[0]), self.X[0]) + np.dot(np.transpose(self.W[1]), self.X[1])

        err = s - y


        # Update fundamental angular frequency
        z = 0
        for i in range(self.n):
            z += (i+1)*(self.W[0][i]*self.X[1][i] - self.W[1][i]*self.X[0][i])
        self.v0 = self.v0 + 2*self.mu0*err*z

        # Update weights
        self.W[0] += 2 * self.mu * self.X[0] * err
        self.W[1] += 2 * self.mu * self.X[1] * err


        self.estimatedFrequency = self.v0 / (2 * math.pi)


        return y

class BMFLC():

    """ FLC filter class

    Attributes
    ----------
    n : int
        Number of harmonics
    X : ndarray
        Reference input vector
    W : ndarray
        Weights
    V : ndarray
        Angular  frequencies
    mu : float
        Adaptive filter gain
    f0 : float
        Starting frequency
    dF : float
        Frequrncey step
    """

    def __init__(self, mu=0.01, fmin=6, fmax=7, dF=0.2):
        """
        Parameters
        ----------
        n : int
            Number of harmonics
        mu : float
            Adaptive filter gain
        f0 : float
            Starting frequency
        """

        self.n = int((fmax - fmin) / dF) + 1
        self.mu = mu
        self.fmax = fmax
        self.fmin = fmin
        self.X = np.zeros(shape=(2, self.n))
        self.W = np.zeros(shape=(2, self.n))
        self.V = np.array(np.zeros([self.n]))
        self.estimatedFrequency = 0

        for i in range(self.n):
            self.V[i] = 2 * math.pi * (self.fmin + dF * i);


    def BMFLC(self, k, s):
        """ BMFLC filter

        Parameters
        ----------
        k : float
            Time instant
        s : float
            Reference signal

        Returns
        -------
        y : float
            Estimated signal
        """
        for i in range(self.n):
            self.X[0][i] = math.sin(self.V[i] * k)
            self.X[1][i] = math.cos(self.V[i] * k)

        y = np.dot(np.transpose(self.W[0]), self.X[0]) + np.dot(np.transpose(self.W[1]), self.X[1])

        err = s - y

        # Update weights
        for i in range(self.n):
            self.W[0][i] += 2 * self.mu * self.X[0][i] * err
            self.W[1][i] += 2 * self.mu * self.X[1][i] * err

        a=0
        b=0
        vest = 0
        for i in range(self.n):
            a += (self.W[0][i]**2+self.W[1][i]**2)*self.V[i]
            for i in range(self.n):
                b += self.W[0][i] ** 2 + self.W[1][i] ** 2
            vest += a/b
            a=0
            b=0
        self.estimatedFrequency = vest/(2*math.pi)

        return y

class BMFLC_new():

    """ FLC filter class

    Attributes
    ----------
    n : int
        Number of harmonics
    X : ndarray
        Reference input vector
    W : ndarray
        Weights
    V : ndarray
        Angular  frequencies
    mu : float
        Adaptive filter gain
    f_min : float
        Minimum frequency
    f_max : float
        Maximum frequency
    dF : float
        Frequency step
    """

    def __init__(self, mu=0.01, f_min=6, f_max=7, dF=0.1, mu0 = 0.000005, dT=0.01, Tp = 2, alpha = 0.67, beta = 10, l = 0.001):
        """
        Parameters
        ----------
        n : int
            Number of harmonics
        mu : float
            Adaptive filter gain
        f_min : float
            Minimum frequency
        f_max : float
            Maximum frequency
        dT : float
            Sampling time in seconds
        Tp : float
            Width of memory window in seconds
        alpha : float
            Minimum amplification gain for memory window
        beta : float
            Multiplier for mu0 learning rate. The learning rate will start on the multiplied
            value of mu0, and decay to it reaches mu0.
        l : float
            Decay constant for mu0. l for lambda.
        """

        self.n = int((f_max - f_min) / dF) + 1
        self.mu = mu
        self.f_max = f_max
        self.f_min = f_min
        self.X = np.zeros(shape=(2, self.n))
        self.W = np.zeros(shape=(2, self.n))
        self.V = np.array(np.zeros([self.n]))
        self.estimatedFrequency = 0
        self.mu0 = mu0*beta
        self.l = l
        for i in range(self.n):
            self.V[i] = 2 * math.pi * (self.f_min + dF * i);
        self.Vref = deepcopy(self.V)

        self.peak1 = 0
        self.peak2 = 0
        self.valley = 0
        self.peak1Pos = -1
        self.peak2Pos = -1
        self.peak1PosPrev = -1
        self.peak2PosPrev = -1
        self.valleyPos = -1

        self.influencePeak1 = 1
        self.influencePeak2 = 1

        self.unstable = False

        self._dFw = (self.V[1] - self.V[0])
        self._mu0lim = mu0

        # Memory window
        delta = (1 / dT) * Tp
        self.rho = (alpha) ** (1 / delta)

        self.decay_magnitude = 0.1

        print()

    def BMFLC(self, k, s):
        """ BMFLC filter

        Parameters
        ----------
        k : float
            Time instant
        s : float
            Reference signal

        Returns
        -------
        y : float
            Estimated signal
        """

        for i in range(self.n):
            self.X[0][i] = math.sin(self.V[i] * k)
            self.X[1][i] = math.cos(self.V[i] * k)
        y = np.dot(np.transpose(self.W[0]), self.X[0]) + np.dot(np.transpose(self.W[1]), self.X[1])

        err = s - y


        #for i in range(self.n):
        #    z = (i + 1) * (self.W[0][i] * self.X[1][i] - self.W[1][i] * self.X[0][i])
        #    self.V[i] = self.V[i] + 2 * self.mu0 * err * z

        ########################## Method for finding peak 1 and 2 and valley

        self.find_peaks_and_valley(err, k)

        ##########################
        print(self.influencePeak1)

        #Update weights
        for i in range(self.n):
            if i == self.peak1Pos:
                self.W[0][i] = self.W[0][i] * self.rho + 2 * self.mu * self.X[0][i] * err * self.influencePeak1
                self.W[1][i] = self.W[1][i] * self.rho + 2 * self.mu * self.X[1][i] * err * self.influencePeak1
            elif i == self.peak2Pos:
                self.W[0][i] = self.W[0][i] * self.rho + 2 * self.mu * self.X[0][i] * err * self.influencePeak2
                self.W[1][i] = self.W[1][i] * self.rho + 2 * self.mu * self.X[1][i] * err * self.influencePeak2
            else:
                self.W[0][i] = self.W[0][i] * self.rho + 2 * self.mu * self.X[0][i] * err
                self.W[1][i] = self.W[1][i] * self.rho + 2 * self.mu * self.X[1][i] * err

        a=0
        b=0
        vest = 0
        for i in range(self.n):
            a += (self.W[0][i]**2+self.W[1][i]**2)*self.V[i]
            for i in range(self.n):
                b += self.W[0][i] ** 2 + self.W[1][i] ** 2
            vest += a/b
            a=0
            b=0
        self.estimatedFrequency = vest/(2*math.pi)

        #Decay learning rate
        self.m0_decay(k)

        return y

    def m0_decay(self, k):
        if self.mu0 >= self._mu0lim :
            self.mu0 *= math.exp(-1 * self.l * k)

    def find_peaks_and_valley(self, err, k):
        positionList = []
        magnitudes = []
        combined_weights_dict = {x: math.sqrt(self.W[0][x] ** 2 + self.W[1][x] ** 2) for x in range(self.n)}
        for key, v in combined_weights_dict.items():
            magnitudes.append(v)
        for i in range(self.n):
            positionList.append(self.V[i] / (2 * math.pi))
        peakMagnitudes = []
        valleyMagnitudes = []
        peaksPos = []
        valleyPositions = []
        magnitudePrev = 0
        magnitudeNow = 0
        magnitudeDiff = 0
        magnitudeDiffPrev = 0


        # Find peaks
        
        for i in range(self.n):
            magnitudeNow = magnitudes[i]
            magnitudeDiff = magnitudeNow - magnitudePrev

            if magnitudeDiff < 0 and magnitudeDiffPrev > 0:
                peakMagnitudes.append(magnitudePrev)
                peaksPos.append(i - 1)
            magnitudeDiffPrev = magnitudeDiff
            magnitudePrev = magnitudeNow
        if len(peakMagnitudes) > 0:
            peak1Mag = 0
            self.peak1Pos = -1
            peak2Mag = -1
            self.peak2Pos = -1
            for i in range(len(peaksPos)):
                if peakMagnitudes[i] > peak1Mag:
                    peak2Mag = peak1Mag
                    self.peak2Pos = self.peak1Pos
                    peak1Mag = peakMagnitudes[i]
                    self.peak1Pos = peaksPos[i]
                elif peakMagnitudes[i] > peak2Mag:
                    peak2Mag = peakMagnitudes[i]
                    self.peak2Pos = peaksPos[i]

        # Reset angle when peak changes
        if self.peak1Pos != self.peak1PosPrev and self.peak1PosPrev > 0:
            self.V[self.peak1PosPrev] = self.Vref[self.peak1PosPrev]
        if self.peak2Pos != self.peak2PosPrev and self.peak2PosPrev > 0:
            self.V[self.peak2PosPrev] = self.Vref[self.peak2PosPrev]

        self.peak1PosPrev = self.peak1Pos
        self.peak2PosPrev = self.peak2Pos

        # Find valley

        magnitudePrev = 0
        magnitudeNow = 0
        magnitudeDiff = 0
        for i in range(self.n):
            magnitudeNow = magnitudes[i]
            magnitudeDiff = magnitudeNow - magnitudePrev

            if magnitudeDiff > 0 and magnitudeDiffPrev < 0:
                valleyMagnitudes.append(magnitudePrev)
                valleyPositions.append(i - 1)
            magnitudeDiffPrev = magnitudeDiff
            magnitudePrev = magnitudeNow

            if len(valleyMagnitudes) > 0:
                valleyMag = 100
                self.valleyPos = -1

                for i in range(len(valleyPositions)):

                    if valleyMagnitudes[i] < valleyMag:
                        if (self.peak1Pos < valleyPositions[i] < self.peak2Pos or self.peak1Pos > valleyPositions[i] > self.peak2Pos):
                            valleyMag = valleyMagnitudes[i]
                            self.valleyPos = valleyPositions[i]
        if len(peakMagnitudes) > 0:


        # Update weights for peaks
            if self.peak1Pos >= 0 :

                z = (self.peak1Pos+1)*(self.W[0][self.peak1Pos]*self.X[1][self.peak1Pos] - self.W[1][self.peak1Pos]*self.X[0][self.peak1Pos])

                self.V[self.peak1Pos] = self.V[self.peak1Pos] + 2 * self.mu0 * err * z


                if self.V[self.peak1Pos+1] - self.V[self.peak1Pos] < 0:
                    self.influencePeak1 = (self._dFw / abs(self.V[self.peak1Pos + 1] - self.V[self.peak1Pos])) * 0.35 + 0.85
                else:
                    self.influencePeak1 = (self._dFw / abs(self.V[self.peak1Pos - 1] - self.V[self.peak1Pos])) * 0.35 + 0.85

                self.peak1 = self.V[self.peak1Pos]

            if len(peakMagnitudes) > 1 and self.peak2Pos >= 0:

                z = (self.peak2Pos + 1) * (self.W[0][self.peak2Pos] * self.X[1][self.peak2Pos] - self.W[1][self.peak2Pos] * self.X[0][self.peak2Pos])

                self.V[self.peak2Pos] = self.V[self.peak2Pos] + 2 * self.mu0 * err * z


                if self.V[self.peak2Pos+1] - self.V[self.peak2Pos] < 0:
                    self.influencePeak2 = (self._dFw / abs(self.V[self.peak2Pos + 1] - self.V[self.peak2Pos])) * 0.35 + 0.85
                else:
                    self.influencePeak2 = (self._dFw / abs(self.V[self.peak2Pos - 1] - self.V[self.peak2Pos])) * 0.35 + 0.85

                self.peak2 = self.V[self.peak2Pos]


    def find_peaks_and_valley_stable(self, err):
        positionList = []
        combined_weights_value_lst = []
        combined_weights_dict = {x: math.sqrt(self.W[0][x] ** 2 + self.W[1][x] ** 2) for x in range(self.n)}
        for key, v in combined_weights_dict.items():
            combined_weights_value_lst.append(v)
        for i in range(self.n):
            positionList.append(self.V[i] / (2 * math.pi))
        peakMagnitudes = []
        peaksPos = []
        lastValue = 0
        valueNow = 0
        difference = 0
        lastDifference = 0
        for i in range(self.n):
            valueNow = combined_weights_value_lst[i]
            difference = valueNow - lastValue

            if difference < 0 and lastDifference > 0:
                peakMagnitudes.append(lastValue)
                peaksPos.append(i - 1)
            lastDifference = difference
            lastValue = valueNow


        if len(peakMagnitudes) > 0:
            peak1Mag = 0
            self.peak1Pos = -1
            peak2Mag = -1
            self.peak2Pos = -1
            for i in range(len(peaksPos)):
                if peakMagnitudes[i] > peak1Mag:
                    peak2Mag = peak1Mag
                    self.peak2Pos = self.peak1Pos
                    peak1Mag = peakMagnitudes[i]
                    self.peak1Pos = peaksPos[i]
                elif peakMagnitudes[i] > peak2Mag:
                    peak2Mag = peakMagnitudes[i]
                    self.peak2Pos = peaksPos[i]


            # Update weights for peaks
            if self.peak1Pos > 0:
                z = (self.peak1Pos + 1) * (
                    self.W[0][self.peak1Pos] * self.X[1][self.peak1Pos] - self.W[1][self.peak1Pos] * self.X[0][
                        self.peak1Pos])
                self.V[self.peak1Pos] = self.V[self.peak1Pos] + 2 * self.mu0 * err * z
                self.peak1 = self.V[self.peak1Pos]
        if len(peakMagnitudes) > 1 and self.peak2Pos >= 0:
            z = (self.peak2Pos + 1) * (
                self.W[0][self.peak2Pos] * self.X[1][self.peak2Pos] - self.W[1][self.peak2Pos] * self.X[0][
                    self.peak2Pos])
            self.V[self.peak2Pos] = self.V[self.peak2Pos] + 2 * self.mu0 * err * z
            self.peak2 = self.V[self.peak2Pos]


class EBMFLC():

    """ FLC filter class

    Attributes
    ----------
    n : int
        Number of harmonics
    X : ndarray
        Reference input vector
    W : ndarray
        Weights
    V : ndarray
        Angular  frequencies
    mu : float
        Adaptive filter gain
    f0 : float
        Starting frequency
    dF : float
        Frequrncey step
    """

    def __init__(self, mu=0.01, fmin=0,fmax=20,fa = 0,fb=2, fc=6, fd=8, dF=0.2, dT=0.01, Tp = 2, alpha = 0.05):
        """
        Parameters
        ----------
        n : int
            Number of harmonics
        mu : float
            Adaptive filter gain
        fmin : float
            Starting frequency of complete frequency range
        fmax : float
            Max frequency of the complete frequency range
        fa : float
            Starting frequency of voluntary motion
        fb : float
            End frequency for voluntary motion
        fc : float
            Start frequency for involuntary motion
        fd : float
            End frequency for involuntary motion
        dT : float
            Sampling time in seconds
        Tp : float
            Width of memory window in seconds
        """

        self.Na = int((fa - 0) / dF)
        self.Nb = int((fb - 0) / dF)
        self.Nc = int((fc - 0) / dF)
        self.Nd = int(round((fd - 0) / dF)) + 1
        self.n = int((fmax - fmin) / dF) + 1
        self.mu = mu
        self.fmax = fmax
        self.fmin = fmin
        self.fa = fa
        self.fb = fb
        self.fc = fc
        self.fd = fd
        self.X = np.zeros(shape=(2, self.n))
        self.Xi = np.zeros(shape=(2, self.Nd - self.Nc))
        self.W = np.zeros(shape=(2, self.n))
        self.Wi = np.zeros(shape=(2, self.Nd - self.Nc))
        self.V = np.array(np.zeros([self.n]))


        delta = (1/dT)*Tp
        self.rho = (alpha)**(1/delta)

        self.estimatedFrequency = 0

        for i in range(self.n):
            self.V[i] = 2 * math.pi * (self.fmin + dF * i);

        self.Vab = self.V[self.Na:self.Nb]
        self.Vcd = self.V[self.Nc:self.Nd]





    def EBMFLC(self,k, s):
        """ BMFLC filter

        Parameters
        ----------
        k : float
            Time instant
        s : float
            Reference signal

        Returns
        -------
        y : float
            Estimated signal
        """
        for i in range(self.n):
            self.X[0][i] = math.sin(self.V[i] * k)
            self.X[1][i] = math.cos(self.V[i] * k)

        m = np.dot(np.transpose(self.W[0]), self.X[0]) + np.dot(np.transpose(self.W[1]), self.X[1])

        err = s - m

        # Update weights
        for i in range(self.n):
            self.W[0][i] = self.W[0][i]*self.rho + 2 * self.mu * self.X[0][i] * err
            self.W[1][i] = self.W[1][i]*self.rho + 2 * self.mu * self.X[1][i] * err

        self.Wi[0] = self.W[0][self.Nc:self.Nd]
        self.Wi[1] = self.W[1][self.Nc:self.Nd]

        self.Xi[0] = self.X[0][self.Nc:self.Nd]
        self.Xi[1] = self.X[1][self.Nc:self.Nd]

        mi = np.dot(np.transpose(self.Wi[0]), self.Xi[0]) + np.dot(np.transpose(self.Wi[1]), self.Xi[1])


        a=0
        b=0
        vest = 0
        for i in range((self.Nd-self.Nc)):
            a += (self.Wi[0][i]**2+self.Wi[1][i]**2)*self.Vcd[i]
            for i in range((self.Nd-self.Nc)):
                b += self.Wi[0][i] ** 2 + self.Wi[1][i] ** 2
            vest += a/b
            a=0
            b=0
        self.estimatedFrequency = vest/(2*math.pi)
        #print(self.estimatedFrequency)



        return mi
