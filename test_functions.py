import torch
import torch.nn.functional as F


class Data:
	def __init__(self,function):
		self.function = function
	def set_input_random(self,randomizer):
		self.input_random = randomizer
	def set_noise_random(self,randomizer):
		self.noise_random = randomizer
	def step(self):
		x = self.input_random()
		y = self.function(x) + self.noise_random(x)
		return(x,y)
	def generator(self):
		while True:
			yield self.step()


class OneDimSystem(Data):
	def __init__(self,interval,function,noise_function):
		super().__init__(function)
		self.a, self.b = interval
		self.noise_function = noise_function
		self.set_input_random(lambda a=self.a,b=self.b: torch.rand(1)*(b-a)+a)
		self.set_noise_random(lambda x: torch.randn(1)*abs(self.noise_function(x)))
		self.recent_points = []
		self.draw_x = torch.linspace(self.a,self.b,10**4)
		self.draw_y = self.function(self.draw_x)
		self.draw_std = abs(self.noise_function(self.draw_x))
	def step(self):
		out = super().step()
		self.recent_points.append(out)
		return(out)
	def draw(self,axes):
		x = self.draw_x.detach().numpy()
		y = self.draw_y.detach().numpy()
		std = self.draw_std.detach().numpy()
		for ax in axes:
			ax.plot(x,y,color='black')
			ax.plot(x,y+std,color='gray',linestyle='dashed')
			ax.plot(x,y-std,color='gray',linestyle='dashed')
	def draw_recent(self,axes,n=10):
		x = [ p[0].item() for p in self.recent_points[-n:] ]
		y = [ p[1].item() for p in self.recent_points[-n:] ]
		for ax in axes:
			ax.scatter(x,y)
	def draw_guess(self,model,axes,color_mean='red',color_std='green'):
		guess = model.predict(self.draw_x.view(-1,1))
		y = guess['guess']
		try:
			std = guess['std']
			std = std.detach().numpy()
		except KeyError:
			pass
		x = self.draw_x.detach().numpy()
		y = y.detach().numpy()
		for ax in axes:
			ax.plot(x,y,color=color_mean)
			try:
				ax.plot(x,y+std,color=color_std,linestyle='dashed')
				ax.plot(x,y-std,color=color_std,linestyle='dashed')
			except NameError:
				pass
	def set_draw_limits(self,axes):
		y = self.draw_y.detach().numpy()
		std = self.draw_std.detach().numpy()
		top = (y+2*std).max()
		bottom = (y-2*std).min()
		for ax in axes:
			ax.set_ylim([bottom,top])


class OneDimVectorData(Data):
	def __init__(self,vector):
		self.vector = vector
		function = lambda i, v=self.vector: v[(i*len(v)).long()]
		super().__init__(function)
		self.noise_function = lambda x: torch.zeros_like(x)
		self.set_input_random(lambda v=self.vector: torch.randint(len(v),(1,))/len(v))
		self.set_noise_random(lambda x: torch.randn(1)*abs(self.noise_function(x)))
		self.recent_points = []
		self.draw_x = torch.arange(len(self.vector))/len(self.vector)
		self.draw_y = self.function(self.draw_x)
		self.draw_std = abs(self.noise_function(self.draw_x))
	def step(self):
		out = super().step()
		self.recent_points.append(out)
		return(out)
	def draw(self,axes):
		x = self.draw_x.detach().numpy()
		y = self.draw_y.detach().numpy()
		for ax in axes:
			ax.plot(x,y,color='black')
	def draw_recent(self,axes,n=10):
		x = [ p[0].item() for p in self.recent_points[-n:] ]
		y = [ p[1].item() for p in self.recent_points[-n:] ]
		for ax in axes:
			ax.scatter(x,y)
	def draw_guess(self,model,axes,color_mean='red',color_std='green'):
		guess = model.predict(self.draw_x.view(-1,1))
		y = guess['guess']
		try:
			std = guess['std']
			std = std.detach().numpy()
		except KeyError:
			pass
		x = self.draw_x.detach().numpy()
		y = y.detach().numpy()
		for ax in axes:
			ax.plot(x,y,color=color_mean)
			try:
				ax.plot(x,y+std,color=color_std,linestyle='dashed')
				ax.plot(x,y-std,color=color_std,linestyle='dashed')
			except NameError:
				pass
	def set_draw_limits(self,axes):
		y = self.draw_y.detach().numpy()
		top = y.max()+1
		bottom = y.min()-1
		for ax in axes:
			ax.set_ylim([bottom,top])

