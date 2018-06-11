import matplotlib
gui_env = ['Qt5Agg','macosx','TKAgg','GTKAgg','Qt4Agg','WXAgg']
gui = gui_env[0]
matplotlib.use(gui,warn=False, force=True)
import serial
import matplotlib.pyplot as plt
import re
import math
import numpy as np
import time
from threading import Thread
import collections

q1 = collections.deque(maxlen=200)
q2 = collections.deque(maxlen=200)
q3 = collections.deque(maxlen=200)

buffer = collections.deque(maxlen=200)

s = serial.Serial(port='/dev/cu.usbmodem1421', baudrate=115200)

s.write(b'text')

p = re.compile("[-+]?\d*\.\d+e*[-+]*\d*|\d+")


start_time = time.time()
plt.ion()
plt.show()

def worker():
    while(True):
        if(len(buffer) != 200):
            buffer.append(s.readline())





t = Thread(target=worker)
t.start()


while(True):

    if(len(buffer) == 200):

        for msg in buffer:
            data = list(map(float, re.findall(p, str(msg))))
            if len(data) == 2:
                q1.appendleft(data[0])
                q2.appendleft(data[1])
                time_now = time.time() - start_time
                q3.appendleft(time_now)
        buffer.clear()
        plt.clf()
        #plt.plot(x[1],x[0])
        #plt.plot(y[1], y[0])
        plt.plot(list(q3), list(q1))
        plt.plot(list(q3), list(q2))
        plt.draw()
        plt.pause(0.000000001)
        time.sleep(0.0001)









