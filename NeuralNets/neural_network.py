import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
	def __init__(self,in_dim,out_dim,width,depth,acti=nn.ReLU()):
		super(NeuralNet,self).__init__()
		layer_dims = [in_dim]+depth*[width]+[out_dim]
		layers = []
		for i in range(depth+1):
			layers += [nn.Linear(layer_dims[i],layer_dims[i+1],bias=True)]
			layers += [acti]
		self.model = nn.Sequential(*layers[:-1])
	def forward(self,x):
		return(self.model(x))
	def set_loss_fn(self,loss_fn):
		self.loss_fn = loss_fn
	def set_optim(self,optim):
		self.optim = optim(self.parameters())
	def train_step(self,in_val,out_val):
		self.optim.zero_grad()
		guess = self.forward(in_val)
		loss = self.loss_fn(guess,out_val)
		loss.backward()
		self.optim.step()


class NormalMeanNet(NeuralNet):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.set_loss_fn(nn.MSELoss())
	def predict(self,x):
		out = self.forward(x)
		return({'guess':out})


class NormalMeanStdNet(NeuralNet):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		loss_fn = lambda guess,target: (guess[0]-target)**2/torch.exp(2*guess[1])+2*guess[1]
		self.set_loss_fn(loss_fn)
	def predict(self,x):
		out = self.forward(x)
		return({'guess':out[:,0],'std':torch.exp(out[:,1])})

