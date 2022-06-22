
import sys

with open(str(sys.argv[1]),'r') as f:
    data = [float(line.rstrip()) for line in f.readlines()]
    f.close()
mean = float(sum(data))/len(data) if len(data) > 0 else float('nan')
print(mean)