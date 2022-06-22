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
header_list = ['window' ,'dropped_frame' ,'accuracy' ,'threshold']
df = pd.read_csv('validate/drop.csv')
if True:
    with open('validate/drop.csv',newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open('validate/drop.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(header_list)
        w.writerows(data)

g = sns.lineplot(data=df, y = 'accuracy', x='dropped_frame',  hue = "window", alpha=0.4)
plt.tight_layout()
plt.savefig("drop.png")
plt.clf()
