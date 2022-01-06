from NeuralNets.neural_network import *
from test_functions import *
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os


test_setups = [
  {'target':lambda x:torch.ones_like(x),
   'noise':lambda x: torch.ones_like(x),
   'interval':(-1,1),
   'name':'Constant Target Constant Noise'
  },
  {'target':lambda x:x, 
   'noise':lambda x: torch.ones_like(x), 
   'interval':(-1,1),
   'name':'Linear Target Constant Noise'
  },
  {'target':lambda x:x**2, 
   'noise':lambda x: torch.ones_like(x), 
   'interval':(-1,1),
   'name':'Square Target Constant Noise'
  },
  {'target':lambda x:torch.sigmoid(3*x), 
   'noise':lambda x: torch.ones_like(x),
   'interval':(-1,1),
   'name':'Sigmoid Target Constant Noise'
  },
  {'target':lambda x:torch.cos(2*np.pi*x),
   'noise':lambda x: torch.ones_like(x),
   'interval':(-1,1),
   'name':'Cosine Target Constant Noise'
  },
  {'target':lambda x:torch.heaviside(x,torch.zeros(1)),
   'noise':lambda x: torch.ones_like(x),
   'interval':(-1,1),
   'name':'Step Target Constant Noise'
  },
  {'target':lambda x: torch.ones_like(x),
   'noise':lambda x: x**2+.1,
   'interval':(-1,1),
   'name':'Constant Target Square Noise'
  },
  {'target':lambda x:x,
   'noise':lambda x: x**2+.1,
   'interval':(-1,1),
   'name':'Linear Target Square Noise'
  },
  {'target':lambda x:x**2,
   'noise':lambda x: x**2+.1,
   'interval':(-1,1),
   'name':'Square Target Square Noise'
  },
  {'target':lambda x:torch.sigmoid(3*x), 
   'noise':lambda x: x**2+.1,
   'interval':(-1,1),
   'name':'Sigmoid Target Square Noise'
  },
  {'target':lambda x:torch.cos(2*np.pi*x), 
   'noise':lambda x: x**2+.1, 
   'interval':(-1,1),
   'name':'Cosine Target Square Noise'
  },
  {'target':lambda x:torch.heaviside(x,torch.zeros(1)), 
   'noise':lambda x: x**2+.1, 
   'interval':(-1,1),
   'name':'Step Target Square Noise'
  },
  {'target':lambda x: torch.ones_like(x), 
   'noise':lambda x: torch.ones_like(x)-x**2+.1, 
   'interval':(-1,1),
   'name':'Constant Target Inverted Square Noise'
  },
  {'target':lambda x:x, 
   'noise':lambda x: torch.ones_like(x)-x**2+.1, 
   'interval':(-1,1),
   'name':'Linear Target Inverted Square Noise'
  },
  {'target':lambda x:x**2,
   'noise':lambda x: torch.ones_like(x)-x**2+.1,
   'interval':(-1,1),
   'name':'Square Target Inverted Square Noise'
  },
  {'target':lambda x:torch.sigmoid(3*x),
   'noise':lambda x: 1.-x**2+.1,
   'interval':(-1,1),
   'name':'Sigmoid Target Inverted Square Noise'
  },
  {'target':lambda x:torch.cos(2*np.pi*x),
   'noise':lambda x: torch.ones_like(x)-x**2+.1, 
   'interval':(-1,1),
   'name':'Cosine Target Inverted Square Noise'
  },
  {'target':lambda x:torch.heaviside(x,torch.zeros(1)),
   'noise':lambda x: torch.ones_like(x)-x**2+.1,
   'interval':(-1,1),
   'name':'Step Target Inverted Square Noise'
  },
  {'target':lambda x: torch.ones_like(x),
   'noise':lambda x: torch.heaviside(x,torch.zeros(1))+.1,
   'interval':(-1,1),
   'name':'Constant Target Step Noise'
  },
  {'target':lambda x:x,
   'noise':lambda x: torch.heaviside(x,torch.zeros(1))+.1,
   'interval':(-1,1),
   'name':'Linear Target Step Noise'
  },
  {'target':lambda x:x**2,
   'noise':lambda x: torch.heaviside(x,torch.zeros(1))+.1,
   'interval':(-1,1),
   'name':'Square Target Step Noise'
  },
  {'target':lambda x:torch.sigmoid(3*x),
   'noise':lambda x: torch.heaviside(x,torch.zeros(1))+.1,
   'interval':(-1,1),
   'name':'Sigmoid Target Step Noise'
  },
  {'target':lambda x:torch.cos(2*np.pi*x),
   'noise':lambda x: torch.heaviside(x,torch.zeros(1))+.1,
   'interval':(-1,1),
   'name':'Cosine Target Step Noise'
  },
  {'target':lambda x:torch.heaviside(x,torch.zeros(1)),
   'noise':lambda x: torch.heaviside(x,torch.zeros(1))+.1,
   'interval':(-1,1),
   'name':'Step Target Step Noise'
  },
]



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
	for setup in test_setups:
		target_function = setup['target']
		noise_function = setup['noise']
		interval = setup['interval']
		name = setup['name']
		data = OneDimSystem(interval,target_function,noise_function)
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


path = os.path.join(res_dir,'results functions with noise'+'.txt')
with open(path,'w') as file:
	print('Approximations, standard net vs net with std, standard net > net with std',file=file)
	for key,value in results.items():
		print('Net size: {0}'.format(key),file=file)
		for key2,value2 in value.items():
			print('\t{0}: {1}, {2}, {3}'.format(key2,value2[0],value2[1],value2[0]>value2[1]),file=file)



















