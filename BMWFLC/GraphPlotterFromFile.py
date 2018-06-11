import matplotlib
gui_env = ['Qt5Agg','macosx','TKAgg','GTKAgg','Qt4Agg','WXAgg']
gui = gui_env[0]
matplotlib.use(gui,warn=False, force=True)
import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('seaborn') #['_classic_test', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn']
import re
import sys


print(matplotlib.matplotlib_fname())

def plot_psd(x,Fs=100):
    plt.subplot(212)
    plt.psd(x,Fs=Fs)
    plt.show()


p = re.compile("[-+]?\d*\.\d+e*[-+]*\d*|\d+")
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
current_palette = sns.color_palette()
#sns.set_palette(current_palette)
plt.title("tremor estimation")

timestep = 0.01
time = 0
while(True):

    f = open("ChFig1.txt", "r")  # opens file with name of "test.txt"

    lines = []
    data = []
    x = []
    y = []
    z = []
    for line in f:
        lines.append(line)


    for msg in lines:
        data = list(map(float, re.findall(p, str(msg))))
        if len(data) == 2:
            x.append(data[0])
            y.append(data[1])
            z.append(time)
            time += timestep
    
    plt.plot(z,x)
    #plt.plot(z, y)
    fig1 = plt.gcf()
    fig1.savefig('test.pdf')
    plt.show()
    sys.exit()







