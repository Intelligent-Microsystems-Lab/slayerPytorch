import pandas as pd
import sys
import matplotlib.pyplot as plt
filename = sys.argv[1]
df = pd.read_csv(filename + '.txt', sep='     ', engine='python')
df.columns = ['Train', 'Test']
x_val = [i for i in range(df['Train'].size)]
plt.plot(x_val, df['Train'])
plt.plot(x_val, df['Test'])
plt.savefig(filename + '.png')
