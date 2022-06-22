import pandas as pd
import numpy as np
import os
import yaml
import sys
import subprocess
import itertools
import time

crc_path = '/afs/crc.nd.edu/user/m/mhoreni/cimflow/timeloop/bin/timeloop-mapper-new'
#This is task ID
my_arg = int(sys.argv[-1]) - 1

network = "sparse"



    vgg = pd.DataFrame(layers, columns = ["weights", "inputs" , "outputs"   ,  "C"  ,   "M"   , "P"   , "Q" , "R" , "S" , "HS", "WS", "Type"])

print(len(list(itertools.product(np.arange(14,15), np.arange(12,13),[16384], [12], [192], ['ruby'], np.arange(0, len(layers))))))

opt, num_pe_x, num_pe_y, gbuff_mem, ifmap_mem, w_pad_mem, constraint_model, layer_id = list(itertools.product(["[' edp ']"], np.arange(14,15), np.arange(12,13),[16384], [12], [192], ['ruby'], np.arange(0, len(layers))))[my_arg]


os.system("mkdir " + network)

os.system("mkdir " + network + "/" + str(my_arg))

os.system("cp -r arch " + network + "/" + str(my_arg) + "/arch")
os.system("cp -r mapper " + network + "/" + str(my_arg) + "/mapper")
os.system("cp -r constraints " + network + "/" + str(my_arg) + "/constraints")
if not ("db" in network):
    os.system("cp -r prob " + network + "/" + str(my_arg) + "/prob")
else:
    os.system("mkdir " + network + "/" + str(my_arg) + "/prob")
    os.system("cp deep_bench/" + os.listdir("deep_bench/")[layers[layer_id]] + " " + network + "/" + str(my_arg) + "/prob/example_layer.yaml")
os.system("cp -r var " + network + "/" + str(my_arg) + "/var")



def set_prob(atts):

def modprob():

    prob = yaml.load(open(network + "/" + str(my_arg) + "/prob/example_layer.yaml", 'r'))

    if constraint_model == "pad" and int(np.ceil(int(prob['problem']['instance']['Q']))) > num_pe_x:
        print("hi")
        prob['problem']['instance']['Q'] = int(np.ceil(int(prob['problem']['instance']['Q'])/num_pe_x) * num_pe_x)
        print(int(np.ceil(int(prob['problem']['instance']['Q'])/num_pe_x) * num_pe_x))
    else:
        prob['problem']['instance']['Q'] = int(prob['problem']['instance']['Q'])

    yaml.dump(prob, open(network + "/" + str(my_arg) + "/prob/example_layer.yaml", 'w'))


arch = yaml.load(open(network + "/" + str(my_arg) + "/arch/eyeriss_like.yaml", 'r'))
my_vars = yaml.load(open(network + "/" + str(my_arg) + "/var/vars.yaml", 'r'))
my_cons = yaml.load(open(network + "/" + str(my_arg) + "/constraints/constraints.yaml", 'r'))

# num_pe_y = 3

#   glb_data_storage_depth: 512 

#   glb_mesh: 8


#   reg_mesh: 32

#   ispad_data_storage_depth: 12

#   wspad_data_storage_depth: 192



numeric_vars = [gbuff_mem, num_pe_x, ifmap_mem, w_pad_mem]


#dummy num string
arch['architecture']['subtree'][0]['subtree'][0]['local'][1]['name'] = arch['architecture']['subtree'][0]['subtree'][0]['local'][1]['name'].replace("13", str(num_pe_x-1))

#PE num string
arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['name'] = arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['name'].replace("167", str(num_pe_x*num_pe_y - 1))



for i,j in enumerate(my_vars['variables'].keys()):
    my_vars['variables'][j] = int(numeric_vars[i])

if constraint_model == "pad":
    my_cons['mapspace_constraints']['template'] = "uber"
else:
    my_cons['mapspace_constraints']['template'] = constraint_model


with open(network + "/" + str(my_arg) + "/mapper/mapper.yaml", 'r') as f:
    mapper = f.read()

mapper = mapper.replace("[ delay, energy ]", opt)

with open(network + "/" + str(my_arg) + "/mapper/mapper.yaml", 'w') as f:
    mapper = f.write(mapper)


yaml.dump(arch, open(network + "/" + str(my_arg) + "/arch/eyeriss_like.yaml", 'w'))
yaml.dump(my_vars, open(network + "/" + str(my_arg) + "/var/vars.yaml", 'w'))
yaml.dump(my_cons, open(network + "/" + str(my_arg) + "/constraints/constraints.yaml", 'w'))

def run_timeloop():
        # proc = subprocess.Popen(["timeloop-mapper", "arch/simba_like_dram.yaml", "arch/components/*.yaml", "prob/cnn-layer.prob.yaml", "mapper/mapper.yaml", "constraints/*.yaml", "sparse/*.yaml"], stdin=subprocess.PIPE, encoding='utf8')
    proc = subprocess.Popen([crc_path, "arch/eyeriss_like.yaml", "arch/components/*.yaml", "mapper/mapper.yaml", "constraints/constraints.yaml", "prob/example_layer.yaml", "var/vars.yaml"], cwd = network + "/" + str(my_arg), stdin=subprocess.PIPE, encoding='utf8')
    try:
        while True:
            proc.stdin.write('f')
    except:
        print("Done")
    proc.wait()


if not ("db" in network):
    set_prob(vgg.iloc[int(layer_id)])

modprob()
done = False
counter = 0


run_timeloop()

