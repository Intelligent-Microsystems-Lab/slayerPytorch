import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import pandas as pd
import torch.nn.utils.prune as prune
import argparse
from itertools import chain

parser = argparse.ArgumentParser(description='PyTorch SLAYER Training')
parser.add_argument('--prunerate', default=0, type=float, metavar='DR',
					help='pruneout') 
parser.add_argument('--droprate', default=0.1, type=float, metavar='DR',
					help='dropout')
parser.add_argument('--validate', default=False, type=bool, metavar='DR',
					help='validate')                    
parser.add_argument('--frame_drope', default=3, type=int, metavar='DR',
					help='number of consecutive frames to drop')                    
parser.add_argument('--threshold', default=20, type=int, metavar='DR',
					help='density threshold for frames to drop')                    
									  
# Define dataset module
class IBMGestureDataset(Dataset):
	def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
		self.path = datasetPath 

		#self.samples = np.loadtxt(sampleFile, skiprows=1).astype('int')
		self.samples = pd.read_csv(sampleFile) 
		self.samplingTime = samplingTime
		self.nTimeBins    = int(sampleLength / samplingTime)

	def __getitem__(self, index):
		# Read inoput and label
		# inputIndex  = self.samples[index, 0]
		# classLabel  = self.samples[index, 1]
		inputIndex  = self.samples['sample'][index]
		classLabel  = self.samples['labels'][index]     
		# Read input spike
		# inputSpikes = snn.io.read2Dspikes(
		#                 self.path + str(inputIndex.item()) + '.bs2'
		#                 ).toSpikeTensor(torch.zeros((2,128,128,self.nTimeBins)),
		#                 samplingTime=self.samplingTime)
		inputSpikes = snn.io.readNpSpikes(
			self.path + str(inputIndex) + '.npy'
		).toSpikeTensor(torch.zeros((2, 128, 128, self.nTimeBins)),
					samplingTime=self.samplingTime)        
		
		# Create one-hot encoded desired matrix
		desiredClass = torch.zeros((11, 1, 1, 1))
		desiredClass[classLabel,...] = 1
		
		return inputSpikes, desiredClass, classLabel

	def __len__(self):
		return self.samples.shape[0]
		
# Define the network
class Network(torch.nn.Module):
	def __init__(self, netParams):
		super(Network, self).__init__()
		# initialize slayer
		slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
		self.slayer = slayer
		# define network functions
		self.conv1 = slayer.conv(2, 16, 5, padding=2, weightScale=10)
		self.conv2 = slayer.conv(16, 32, 3, padding=1, weightScale=50)
		self.pool1 = slayer.pool(4)
		self.pool2 = slayer.pool(2)
		self.pool3 = slayer.pool(2)
		self.fc1   = slayer.dense((8*8*32), 512)
		self.fc2   = slayer.dense(512, 11)
		self.drop  = slayer.dropout(args.droprate)
		self.conv1_dropped_frames  = []
		self.conv2_dropped_frames  = []
		self.dense1_dropped_frames = []
		self.dropped_frames = []
		self.threshold = []
		# self.conv1_frame_density_log = []
		# self.conv2_frame_density_log = [] 
		# self.dense1_frame_density_log= []
	def forward(self, spikeInput,threshold):
		self.threshold = threshold
		spike = self.slayer.spikeLoihi(self.pool1(spikeInput )) # 32, 32, 2
		spike = self.slayer.delayShift(spike, 1)
		frame_counter = 0
		dropped_frames_counter = 0
		for batch in range(4):
			for timestep in range(1450):
				frameDensity = np.count_nonzero(spike[batch,:,:,:,timestep].cpu().detach().numpy())
				if frameDensity < self.threshold[0]:
					frame_counter+=1
				else:
					frame_counter = 0
				if frame_counter == args.frame_drope:
					dropped_frames_counter +=args.frame_drope
					while frame_counter:
						frame_counter -= 1
						spike[batch,:,:,:,timestep-frame_counter] = 0
		
		self.conv1_dropped_frames.append(dropped_frames_counter)                
		#with open('validate/log.txt', 'a') as f:
		#    f.write(str(dropped_frames)+ '\n')                   
		#print('Number of frames dropped: ' + str(dropped_frames))
		#spike = self.drop(spike)
		spike = self.slayer.spikeLoihi(self.conv1(spike)) # 32, 32, 16
		spike = self.slayer.delayShift(spike, 1)
		
		spike = self.slayer.spikeLoihi(self.pool2(spike)) # 16, 16, 16
		spike = self.slayer.delayShift(spike, 1)
		#Dropping frames for Conv2 input activations
		for batch in range(4):
			for timestep in range(1450):
				frameDensity = np.count_nonzero(spike[batch,:,:,:,timestep].cpu().detach().numpy())
				if frameDensity < self.threshold[1]:
					frame_counter+=1
				else:
					frame_counter = 0
				if frame_counter == args.frame_drope:
					dropped_frames_counter +=args.frame_drope
					while frame_counter:
						frame_counter -= 1
						spike[batch,:,:,:,timestep-frame_counter] = 0
		#self.conv2_dropped_frames.append(dropped_frames_counter)
		#spike = self.drop(spike)
		spike = self.slayer.spikeLoihi(self.conv2(spike)) # 16, 16, 32
		spike = self.slayer.delayShift(spike, 1)
		
		spike = self.slayer.spikeLoihi(self.pool3(spike)) #  8,  8, 32
		spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))
		spike = self.slayer.delayShift(spike, 1)
		frame_counter = 0
		#dropped_frames_counter = 0

		for batch in range(4):
			for timestep in range(1450):
				frameDensity = np.count_nonzero(spike[batch,:,:,:,timestep].cpu().detach().numpy())
				if frameDensity < self.threshold[2]:
					frame_counter+=1
				else:
					frame_counter = 0
				if frame_counter == args.frame_drope:
					dropped_frames_counter +=args.frame_drope
					while frame_counter:
						frame_counter -= 1
						spike[batch,:,:,:,timestep-frame_counter] = 0
		#self.dense1_dropped_frames.append(dropped_frames_counter) 
		self.dropped_frames.append(dropped_frames_counter)
		#spike = self.drop(spike)
		spike = self.slayer.spikeLoihi(self.fc1  (spike)) # 512
		spike = self.slayer.delayShift(spike, 1)
		
		spike = self.slayer.spikeLoihi(self.fc2  (spike)) # 11
		spike = self.slayer.delayShift(spike, 1)
		
		return spike
		
# Define Loihi parameter generator
def genLoihiParams(net, filename):
	fc1Weights   = snn.utils.quantize(net.fc1.weight  , 2).flatten().cpu().data.numpy()
	fc2Weights   = snn.utils.quantize(net.fc2.weight  , 2).flatten().cpu().data.numpy()
	conv1Weights = snn.utils.quantize(net.conv1.weight, 2).flatten().cpu().data.numpy()
	conv2Weights = snn.utils.quantize(net.conv2.weight, 2).flatten().cpu().data.numpy()
	pool1Weights = snn.utils.quantize(net.pool1.weight, 2).flatten().cpu().data.numpy()
	pool2Weights = snn.utils.quantize(net.pool2.weight, 2).flatten().cpu().data.numpy()
	pool3Weights = snn.utils.quantize(net.pool3.weight, 2).flatten().cpu().data.numpy()
	
	np.save(filename+ '/fc1.npy'  , fc1Weights)
	np.save(filename+ '/fc2.npy'  , fc2Weights)
	np.save(filename+ '/conv1.npy', conv1Weights)
	np.save(filename+ '/conv2.npy', conv2Weights)
	np.save(filename+ '/pool1.npy', pool1Weights)
	np.save(filename+ '/pool2.npy', pool2Weights)
	np.save(filename+ '/pool3.npy', pool3Weights)

	plt.figure(11)
	plt.hist(fc1Weights  , 256)
	plt.title('fc1 weights')

	plt.figure(12)
	plt.hist(fc2Weights  , 256)
	plt.title('fc2 weights')

	plt.figure(13)
	plt.hist(conv1Weights, 256)
	plt.title('conv1 weights')

	plt.figure(14)
	plt.hist(conv2Weights, 256)
	plt.title('conv2 weights')

	plt.figure(15)
	plt.hist(pool1Weights, 256)
	plt.title('pool1 weights')

	plt.figure(16)
	plt.hist(pool2Weights, 256)
	plt.title('pool2 weights')

	plt.figure(17)
	plt.hist(pool3Weights, 256)
	plt.title('pool3 weights')
	
if __name__ == '__main__':
	
	global args
	args = parser.parse_args()
	prunerate = args.prunerate
	droprate = args.droprate
	ispruning = False
	if prunerate !=0:
		ispruning = True    
	netParams = snn.params('network.yaml')
	
	# Define the cuda device to run the code on.
	device = torch.device('cuda')
	# deviceIds = [2, 3]

	# Create network instance.
	net = Network(netParams).to(device)
	# net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)
	''' Uncomment for fine-tuning or validation '''
	net.load_state_dict(torch.load('TrainedFull/ibmGestureNet.pt',map_location=device))
	# Create snn loss instance.
	error = snn.loss(netParams, snn.loihi).to(device)

	# Define optimizer module.
	# optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
	optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.01, amsgrad = True)
	# Dataset and dataLoader instances.
	trainingSet = IBMGestureDataset(datasetPath =netParams['training']['path']['trainFile'], 
									sampleFile  =netParams['training']['path']['train'],
									samplingTime=netParams['simulation']['Ts'],
									sampleLength=netParams['simulation']['tSample'])
	trainLoader = DataLoader(dataset=trainingSet, batch_size=4, shuffle=True, num_workers=1)

	testingSet = IBMGestureDataset(datasetPath  =netParams['training']['path']['testFile'], 
								   sampleFile  =netParams['training']['path']['test'],
								   samplingTime=netParams['simulation']['Ts'],
								   sampleLength=netParams['simulation']['tSample'])
	testLoader = DataLoader(dataset=testingSet, batch_size=4, shuffle=True, num_workers=1)

	# Learning stats instance.
	stats = snn.utils.stats()
	if ispruning:
		for name, module in net.named_modules():
			if 'conv' in name or 'fc1' in name:
				prune.ln_structured(module, name="weight", amount=prunerate, n=1, dim=0)
				prune.remove(module, 'weight')      
	# Visualize the input spikes (first five samples).

	for i in range(5):
		input, target, label = trainingSet[i]
		snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 128, 128, -1)).cpu().data.numpy()))
	# Setting the threshold for the whole training set

	# Just to read the threshold
	threshold = []
	with open('validate/threshold_trainset/conv1_activation_density_log.csv') as handle:
		reader = csv.reader(handle)
		lst = []
		for row in reader:
			lst.append([int(densities) for densities in row])
		next(reader, None)
		flatten = list(chain.from_iterable(lst))
	threshold.append( np.percentile(flatten, args.threshold))
	with open('validate/threshold_trainset/conv2_activation_density_log.csv') as handle:
		reader = csv.reader(handle)
		lst = []
		for row in reader:
			lst.append([int(densities) for densities in row])
		next(reader, None)
		flatten = list(chain.from_iterable(lst))
	threshold.append(np.percentile(flatten, args.threshold))
	with open("./validate/threshold_trainset/dense1_activation_density_log.csv",'r') as handle:
		reader = csv.reader(handle)
		lst = []
		for row in reader:
			lst.append([int(densities) for densities in row])
		next(reader, None)
		flatten = list(chain.from_iterable(lst))
	threshold.append(np.percentile(flatten, args.threshold))		

	# for epoch in range(500):
	for epoch in range(1):
		tSt = datetime.now()
		if not args.validate:     
		# Training loop.
			for i, (input, target, label) in enumerate(trainLoader, 0):
				net.train()

				# Move the input and target to correct GPU.
				input  = input.to(device)
				target = target.to(device) 

				# Forward pass of the network.
				output = net.forward(input,threshold)

				# Gather the training stats.
				stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
				stats.training.numSamples     += len(label)

				# Calculate loss.
				loss = error.numSpikes(output, target)

				# Reset gradients to zero.
				optimizer.zero_grad()

				# Backward pass of the network.
				loss.backward()

				# Update weights.
				optimizer.step()

				# Gather training loss stats.
				stats.training.lossSum += loss.cpu().data.item()

				# Display training stats.
				stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

		# Testing loop.
		# Same steps as Training loops except loss backpropagation and weight update.

		

		for i, (input, target, label) in enumerate(testLoader, 0):
			#net.eval()  #because I want dropping during inference phase and net.eval() prevents it
			with torch.no_grad():
				input  = input.to(device)
				target = target.to(device) 
			output = net.forward(input,threshold)

			stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
			stats.testing.numSamples     += len(label)

			loss = error.numSpikes(output, target)
			stats.testing.lossSum += loss.cpu().data.item()
			stats.print(epoch, i)
		# Update stats.
		stats.update()
		#filename = str('Trained/prunerate_' + str(args.prunerate)+ 'droprate_' + str(args.droprate))
		filename=str('validate/threshold_trainset/dropout_' + str(args.droprate))
		if not os.path.exists(filename):
			os.mkdir(filename)
		stats.plot(saveFig=True, path=filename + '/')
		# if stats.training.bestLoss is True: torch.save(net.state_dict(), filename + '/ibmGestureNet.pt')

	# Save training data
	stats.save(filename + str('/'))
	# net.load_state_dict(torch.load(filename + '/ibmGestureNet.pt'))
	# genLoihiParams(net,filename)
	
	# # Plot the results.
	# # Learning loss
	# plt.figure(1)
	# plt.semilogy(stats.training.lossLog, label='Training')
	# plt.semilogy(stats.testing .lossLog, label='Testing')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.legend()
	
		# Learning accuracy
#   plt.figure(2)
#   plt.plot(stats.training.accuracyLog, label='Training')
#   plt.plot(stats.testing .accuracyLog, label='Testing')
#   plt.xlabel('Epoch')
#   plt.ylabel('Accuracy')
#   plt.legend()
#   plt.show()
	print( 'Accuracy with ' + str(args.frame_drope) + ' consec frame drop:  ' + str(stats.testing.accuracyLog))
	data = [args.frame_drope ,round(sum(net.dropped_frames)/len(net.dropped_frames),2), round(stats.testing.accuracyLog[0],5)*100,  args.threshold]
	header = ['layer', 'window' ,'dropped_frame' ,'accuracy' ,'threshold']
	with open('validate/threshold_trainset/all_layer_drop.csv', 'a', encoding='UTF8') as f:
		writer = csv.writer(f)
		writer.writerow(data)
