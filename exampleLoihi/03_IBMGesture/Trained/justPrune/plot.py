import pandas as pd
import sys
import matplotlib.pyplot as plt
filename = sys.argv[1]
fig = plt.figure()
ax = fig.add_subplot(111)
df = pd.read_csv(filename + '.txt', sep='     ', engine='python')
df.columns = ['Train', 'Test']
x_val = [i for i in range(df['Train'].size)]
ax.plot(x_val, df['Train'], label='50%(Train)')
ax.plot(x_val, df['Test'], label='50%(Test)')
df = pd.read_csv('0.6accuracy' +'.txt', sep='     ', engine='python')
df.columns = ['Train', 'Test']
x_val = [i for i in range(df['Train'].size)]
plt.plot(x_val, df['Train'], label='60%(Train)')
plt.plot(x_val, df['Test'], label='60%(Test)')

df = pd.read_csv('0.8accuracy' +'.txt', sep='     ', engine='python')
df.columns = ['Train', 'Test']
x_val = [i for i in range(df['Train'].size)]
plt.plot(x_val, df['Train'], label='80%(Train)')
plt.plot(x_val, df['Test'], label='80%(Test)')

df = pd.read_csv('0.9accuracy' +'.txt', sep='     ', engine='python')
df.columns = ['Train', 'Test']
x_val = [i for i in range(df['Train'].size)]
plt.plot(x_val, df['Train'], label='90%(Train)')
plt.plot(x_val, df['Test'], label='90%(Test)')

df = pd.read_csv('0.95accuracy' +'.txt', sep='     ', engine='python')
df.columns = ['Train', 'Test']
x_val = [i for i in range(df['Train'].size)]
plt.plot(x_val, df['Train'], label='950%(Train)')
plt.plot(x_val, df['Test'], label='95%(Test)')

leg1 = ax.legend(loc='lower right')
ax.add_artist(leg1)
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.savefig(filename + '.png')
