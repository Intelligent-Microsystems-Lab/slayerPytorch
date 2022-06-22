import subprocess
import sys, os
import pandas as pd
import numpy as np
import shutil
import itertools

# Parent Directory path
path = "/afs/crc.nd.edu/user/p/ptaheri/Private/benchmarkSNN/slayerPytorch/exampleLoihi/03_IBMGesture"

#taskID = int(os.getenv("SGE_TASK_ID"))-1
taskID = int(sys.argv[-1]) - 1
drop , threshold = list(itertools.product(np.arange(2,10), [10,15,20,25,30,35,40,45,50]))[taskID]
proc = subprocess.Popen(["python3 multiFrameRemoval.py --validate True --frame_drope " + str(drop) + ' --threshold ' + str(threshold)], shell = True, stdin=subprocess.PIPE, encoding='utf8', cwd=path)
try:
    while True:
        proc.stdin.write('f')
except:
    print("Done")
proc.wait()
