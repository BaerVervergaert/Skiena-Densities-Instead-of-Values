from NeuralNets.neural_network import *
from test_functions import *
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os


fig,axes = plt.subplots(2,2)
axes = axes.flatten()
plt.ion()

total = 10**5
N=10**3
results = {}
network_shapes = [(10,3),(100,10)]
res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'NeuralNets','Results')
for (width,depth) in network_shapes:
	results[(width,depth)] = {}
	for i in range(10):
		vector = (torch.randn(10000)/100).cumsum(dim=0)
		name = 'Brownian Motion Test {0}'.format(i+1)
		data = OneDimVectorData(vector)
		mean_net = NormalMeanNet(1,1,width,depth)
		meanstd_net = NormalMeanStdNet(1,2,width,depth)
		mean_net_loss = []
		meanstd_net_loss = []
		loss_fn = nn.MSELoss()
		optim_fn = lambda x: optim.Adam(x,amsgrad=True)
		mean_net.set_optim(optim_fn)
		meanstd_net.set_optim(optim_fn)
		for count,d in enumerate(data.generator()):
			if count%total==total-1:
				break
			if count%N==N-1:
				for ax in axes:
					ax.clear()
				data.draw(axes[1:])
				data.set_draw_limits(axes[1:])
				data.draw_recent(axes[1:],n=20)
				data.draw_guess(mean_net,[axes[2]])
				data.draw_guess(meanstd_net,[axes[3]])
				actual_loss = loss_fn(mean_net(data.draw_x.view(-1,1)).view(-1),data.draw_y)
				mean_net_loss.append(np.log(actual_loss.item()))
				actual_loss = loss_fn(meanstd_net(data.draw_x.view(-1,1))[:,0],data.draw_y)
				meanstd_net_loss.append(np.log(actual_loss.item()))
				axes[0].plot(mean_net_loss,color='red',label='Mean only')
				axes[0].plot(meanstd_net_loss,color='blue',label='Mean and Std')
				axes[0].legend()
				fig.suptitle(name)
				axes[0].set_title('Log actual loss')
				axes[1].set_title('Target Function and Data')
				axes[2].set_title('Standard Net')
				axes[3].set_title('Standard Net with Std')
				plt.pause(.01)
			mean_net.train_step(*d)
			meanstd_net.train_step(*d)
		results[(width,depth)][name] = (mean_net_loss[-1],meanstd_net_loss[-1])
		path = os.path.join(res_dir,str((width,depth))+name.strip(' ')+'.png')
		fig.savefig(path)


path = os.path.join(res_dir,'results brownian motion test'+'.txt')
with open(path,'w') as file:
	print('Approximations, standard net vs net with std, standard net > net with std',file=file)
	for key,value in results.items():
		print('Net size: {0}'.format(key),file=file)
		for key2,value2 in value.items():
			print('\t{0}: {1}, {2}, {3}'.format(key2,value2[0],value2[1],value2[0]>value2[1]),file=file)



















