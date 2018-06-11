import math
import numpy as np



def complexSignal(f1=6, f2=7, a1=300, a2 =400, data_points = 3000, dT = 0.01, noisy = True,return_separated_signals = False):
    mean = 0
    std = 10
    if noisy:
        noise = np.random.normal(mean, std, size=data_points)
    else:
        noise = np.zeros(shape=(data_points))
    tremor1 = []
    tremor2 = []
    frequencies = np.zeros(shape=(2, data_points))

    t = 0

    for i in range(data_points):
        t += dT
        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))
        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1
    if return_separated_signals:
        return tremor1, tremor2, noise
    else:
        return np.array(tremor1) + np.array(tremor2) + noise, frequencies

    #return 3.5 * math.sin(2 * math.pi * f1 * t) + 2.5 * math.cos(2 * math.pi * f2 * t)

def signal_action_tremor(f1=4.6, f2=5.2, f3=0.6 ,a1=300, a2 =400, a3 = 800  , data_points = 1000, dT = 0.01, noisy = True,return_separated_signals = False):
    mean = 0
    std = 10
    if noisy:
        noise = np.random.normal(mean, std, size=data_points)
    else:
        noise = np.zeros(shape=(data_points))
    tremor1 = []
    tremor2 = []
    tremor3 = []
    frequencies = np.zeros(shape=(2, data_points))

    t = 0

    for i in range(data_points):
        t += dT
        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))
        tremor3.append(a3 * math.cos(2 * math.pi * f3 * t))
        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1

    if return_separated_signals:
        return tremor1, tremor2, tremor3,noise
    else:
        return np.array(tremor1) + np.array(tremor2) + np.array(tremor3) + noise, frequencies

    #return 3.5 * math.sin(2 * math.pi * f1 * t) + 2.5 * math.cos(2 * math.pi * f2 * t)

def signal_1(f1=6, f2=6.3, a1=200, a2 =400, data_points = 1000, dT = 0.01, return_separated_signals = False):
    mean = 0
    std = 10
    noise = np.random.normal(mean, std, size=data_points)
    tremor1 = []
    tremor2 = []
    frequencies = np.zeros(shape=(2, data_points))

    t = 0

    for i in range(data_points):
        t += dT

        if i >= 300 and i < 500 :
            f2 += 0.0005
            f1 += 0.0005

        if i >= 700 and i < 900:
            f2 -= 0.0005
            f1 -= 0.0005


        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))
        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1

    if return_separated_signals:
        return tremor1, tremor2, noise
    else:
        return np.array(tremor1) + np.array(tremor2) + noise, frequencies

    #return 3.5 * math.sin(2 * math.pi * f1 * t) + 2.5 * math.cos(2 * math.pi * f2 * t)

def signal_freq(f1=6, f2=6.3, a1=200, a2 =400, data_points = 1000, dT = 0.01, return_separated_signals = False):
    mean = 0
    std = 10
    noise = np.random.normal(mean, std, size=data_points)
    tremor1 = []
    tremor2 = []
    frequencies = np.zeros(shape=(2, data_points))

    t = 0

    for i in range(data_points):
        t += dT

        if i == 500:
            f1 += 0.5
        if i == 1000:
            f2 += 0.5
        if i == 2000:
            f1 -= 1
            f2 -= 0.5

        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1




        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))


    if return_separated_signals:
        return tremor1, tremor2, noise
    else:
        return np.array(tremor1) + np.array(tremor2) + noise, frequencies

    #return 3.5 * math.sin(2 * math.pi * f1 * t) + 2.5 * math.cos(2 * math.pi * f2 * t)

def signal_amp(f1=6, f2=6.3, a1=200, a2 =400, data_points = 1000, dT = 0.01, return_separated_signals = False):
    mean = 0
    std = 10
    noise = np.random.normal(mean, std, size=data_points)
    tremor1 = []
    tremor2 = []
    frequencies = np.zeros(shape=(2, data_points))

    t = 0

    for i in range(data_points):
        t += dT

        if i == 500:
            a1 = 500
        if i == 1000:
            a2 = 1000
        if i == 2000:
            a1 = 20
            a2 = 10

        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1

        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))


    if return_separated_signals:
        return tremor1, tremor2, noise
    else:
        return np.array(tremor1) + np.array(tremor2) + noise, frequencies

def signal_amp_freq(f1=6, f2=6.3, a1=200, a2 =400, data_points = 1000, dT = 0.01, return_separated_signals = False):
    mean = 0
    std = 10
    noise = np.random.normal(mean, std, size=data_points)
    tremor1 = []
    tremor2 = []
    frequencies = np.zeros(shape=(2, data_points))

    t = 0

    for i in range(data_points):
        t += dT

        if i == 500:
            a1 = 500
        if i == 1000:
            a2 = 1000
        if i == 2000:
            a1 = 20
            a2 = 10

        if i == 500:
            f1 += 0.5
        if i == 1000:
            f2 += 0.5
        if i == 2000:
            f1 -= 1
            f2 -= 0.5



        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))
        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1

    if return_separated_signals:
        return tremor1, tremor2, noise
    else:
        return np.array(tremor1) + np.array(tremor2) + noise, frequencies


def signal_multiple(f1=6, f2=7, a1=300, a2 =400, data_points = 3000, dT = 0.01, noisy = True,return_separated_signals = False):
    mean = 0
    std = 10
    if noisy:
        noise = np.random.normal(mean, std, size=data_points)
    else:
        noise = np.zeros(shape=(data_points))
    a1 = 100
    a2 = 150
    a3 = 200
    a4 = 80
    a5 = 310
    a6 = 230

    f1 = 4.05
    f2 = 4.55
    f3 = 5.85
    f4 = 8.15
    f5 = 10.25
    f6 = 18.55



    tremor1 = []
    tremor2 = []
    tremor3 = []
    tremor4 = []
    tremor5 = []
    tremor6 = []
    frequencies = np.zeros(shape=(10, data_points))

    t = 0

    for i in range(data_points):
        t += dT
        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))
        tremor3.append(a3 * math.sin(2 * math.pi * f3 * t))
        tremor4.append(a4 * math.cos(2 * math.pi * f4 * t))
        tremor5.append(a5 * math.sin(2 * math.pi * f5 * t))
        tremor6.append(a6 * math.cos(2 * math.pi * f6 * t))
        frequencies[0][i] = f5
        frequencies[1][i] = f6
        frequencies[2][i] = f3
        frequencies[3][i] = f2
        frequencies[4][i] = f1
        frequencies[5][i] = f4

    if return_separated_signals:
        return tremor1, tremor2, tremor3, tremor4, tremor5, tremor6,  noise
    else:
        return np.array(tremor1) + np.array(tremor2) + np.array(tremor3) + np.array(tremor4) + np.array(tremor5) + np.array(tremor6) + noise, frequencies

    #return 3.5 * math.sin(2 * math.pi * f1 * t) + 2.5 * math.cos(2 * math.pi * f2 * t)

def signal_multiple(f1=6, f2=7, a1=300, a2 =400, data_points = 3000, dT = 0.01, noisy = True,return_separated_signals = False):
    mean = 0
    std = 10
    if noisy:
        noise = np.random.normal(mean, std, size=data_points)
    else:
        noise = np.zeros(shape=(data_points))
    a1 = 100
    a2 = 150
    a3 = 200
    a4 = 80
    a5 = 310
    a6 = 230

    f1 = 4.05
    f2 = 4.55
    f3 = 5.85


    tremor1 = []
    tremor2 = []
    tremor3 = []
    tremor4 = []
    tremor5 = []
    tremor6 = []
    frequencies = np.zeros(shape=(10, data_points))

    t = 0

    for i in range(data_points):
        t += dT
        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))
        tremor3.append(a3 * math.sin(2 * math.pi * f3 * t))
        tremor4.append(a4 * math.cos(2 * math.pi * f4 * t))
        tremor5.append(a5 * math.sin(2 * math.pi * f5 * t))
        tremor6.append(a6 * math.cos(2 * math.pi * f6 * t))
        frequencies[0][i] = f5
        frequencies[1][i] = f6
        frequencies[2][i] = f3
        frequencies[3][i] = f2
        frequencies[4][i] = f1
        frequencies[5][i] = f4

    if return_separated_signals:
        return tremor1, tremor2, tremor3, tremor4, tremor5, tremor6,  noise
    else:
        return np.array(tremor1) + np.array(tremor2) + np.array(tremor3) + np.array(tremor4) + np.array(tremor5) + np.array(tremor6) + noise, frequencies

    #return 3.5 * math.sin(2 * math.pi * f1 * t) + 2.5 * math.cos(2 * math.pi * f2 * t)

def signal_3(f1=6, f2=7, a1=200, a2 =400, data_points = 2000, dT = 0.01, return_separated_signals = False):
    mean = 0
    std = 1
    noise = np.random.normal(mean, std, size=data_points)
    tremor1 = []
    tremor2 = []
    frequencies = np.zeros(shape=(2, data_points))

    t = 0

    for i in range(data_points):
        t += dT

        if i >= 300 and i < 2000 :
            f2 += 0.0005
            f1 += 0.0005



        tremor1.append(a1*math.sin(2 * math.pi * f1 * t))
        tremor2.append(a2* math.cos(2 * math.pi * f2 * t))
        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1

    if return_separated_signals:
        return tremor1, tremor2, noise
    else:
        return np.array(tremor1) + np.array(tremor2) + noise, frequencies

    #return 3.5 * math.sin(2 * math.pi * f1 * t) + 2.5 * math.cos(2 * math.pi * f2 * t)