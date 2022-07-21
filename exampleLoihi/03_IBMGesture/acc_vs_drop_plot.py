from turtle import color
from numpy import sort
import yaml
import subprocess
import numpy as np
import sys 
import pandas as pd
import matplotlib.font_manager
import csv
import numpy as np
import os
import matplotlib.patheffects as pe
import seaborn as sns 
import matplotlib.pyplot as plt 
''' This is a plotter for Accuracy vs frame dropps and dropping threshold
    '''
header_list = ['window' ,'dropped_frame' ,'accuracy' ,'threshold']
# df = pd.read_csv('validate/conv2_layer_drop.csv')
if False:
    with open('validate/conv2_layer_drop.csv',newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open('validate/conv2_layer_drop.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(header_list)
        w.writerows(data)
# df['dropped_frame'] = df['dropped_frame'].astype(float)/(1450*4*2/100)
# df['window'] = df['window'].astype(str)
# g = sns.lineplot(data=df, y = 'accuracy', x = 'dropped_frame', alpha=0.4)
df = pd.read_csv('validate/all_layers_drop.csv')
df['dropped_frame'] = df['dropped_frame'].astype(float)/(1450*4*3/100)
df['window'] = df['window'].astype(str)
#g = sns.lineplot(data=df, y = 'accuracy', x = 'dropped_frame', alpha=0.4)
# df = pd.read_csv('validate/drop.csv')
# df['dropped_frame'] = df['dropped_frame'].astype(float)/(1450*4/100)
# df['window'] = df['window'].astype(str)
g = sns.lineplot(data=df, y = 'accuracy', x = 'dropped_frame', alpha=0.4)
maxim = np.max(df['accuracy'])
g.axhline(maxim, ls='--')
import matplotlib.transforms as transforms
trans = transforms.blended_transform_factory(
    g.get_yticklabels()[0].get_transform(), g.transData)
g.text(0,maxim, "{:.2f}".format(maxim), color="red", transform=trans, 
        ha="right", va="center")
#g = sns.barplot(data=df, y = 'accuracy', x = 'window', alpha=0.4)
#second_ax = plt.twinx()
#g = sns.barplot(data=df, y = 'dropped_frame', x = 'window')
plt.tight_layout()
plt.xlabel("Dropped frames(%)")
plt.ylabel("Accuracy(%)")
plt.savefig("acc_drop.png")
plt.clf()
