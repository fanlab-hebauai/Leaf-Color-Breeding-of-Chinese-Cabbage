import pandas as pd
import numpy as np
import os
def smooth(csv_path,weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
 
 
    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    save.to_csv('smooth_'+csv_path)
 
 
if __name__=='__main__':
    smooth('loss.xlsx')

    

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
 
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2])) 
        x.append((row[1]))
    return x ,y
 
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
 
plt.figure()
 
x,y=readcsv("smooth_loss.csv")
plt.plot(x, y, 'r',label='G ')
 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
 
plt.ylim(0, 16)                   # y轴的最大值
plt.xlim(0, 104800)               # x轴最大值
plt.title('loss', fontsize=16) 
plt.xlabel('Steps',fontsize=20)
plt.ylabel('Score',fontsize=20)
plt.legend(fontsize=16)
plt.show()