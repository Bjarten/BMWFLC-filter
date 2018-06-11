import matplotlib
gui_env = ['Qt5Agg','macosx','TKAgg','GTKAgg','Qt4Agg','WXAgg']
gui = gui_env[0]
matplotlib.use(gui,warn=False, force=True)
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use('seaborn') #['_classic_test', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn']
import numpy as np
sns.set_style("whitegrid")
sns.axes_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

# create data
x = [1,3,5,7,9]
values = [1, 1/3, 1/5, 1/7, 1/9]

# stem function: first way
#plt.stem(x, values)
#plt.ylim(0, 1.1)
#plt.show()

# stem function: If no X provided, a sequence of numbers is created by python:
#plt.stem(values)
#plt.show()

# stem function: second way
(markerline, stemlines, baseline) = plt.stem(x, values, markerfmt = 'ro')
plt.setp(baseline, visible=False)
plt.setp(markerline, zorder=3)
plt.ylabel("Magnitude")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 10)
plt.xticks([1,3,5,7,9])
plt.show()

