import torch
import numpy as np

import visdom
import time



VIS = visdom.Visdom()
ENV = 'SOM'


# dim is the deature dimension 
# e.g dim = 0
# X is M x N
# Y is M x K
def pdist2(X, Y, dim):
	N = X.shape[abs(dim-1)]	
	K = Y.shape[abs(dim-1)]
	XX = (X ** 2).sum(dim).unsqueeze(abs(0-dim)).expand(K, -1)
	YY = (Y ** 2).sum(dim).unsqueeze(abs(1-dim)).expand(-1, N)
	if dim == 0:
		XX = XX.expand(K, -1)
		YY = YY.expand(-1, N)
		prod = torch.mm(X.transpose(0,1), Y)
	else:
		XX = XX.expand(-1, K)
		YY = YY.expand(N, -1)
		prod = torch.mm(X, Y.transpose(0,1))
	return XX + YY - 2 * prod


def ind2sub(idx, cols):
	r = idx / cols
	c = idx - r * cols
	return r, c	


def sub2ind(r, c, cols):
	idx = r * cols + c
	return idx	


def generateSphereSurface(N):
	data = torch.randn(N, 3)
	data /= torch.norm(data, 2, 1, True)
	return data

def generateCubeVolume(N):
	return torch.rand(N, 3)

def generateSquareArea(N):
	return torch.rand(N, 2)

def generateCirclePerimeter(N):
	data = torch.randn(N, 2)
	data /= torch.norm(data, 2, 1, True)
	return data

def generateCubeSurface(N, dim):
	side = (torch.rand(N) * 6).floor()
	perp_axis = s % 3





class SOM(object):

	def __init__(self, rows=4, cols=4, dim=3):
		self.rows = rows
		self.cols = cols
		self.dim = dim
		self.contents = None
		self.grid = None
		self.grid_dists = None



	def initialize(self):
		# Randomly initialize unit contents
		self.contents = torch.normal(
			mean=0.0, 
			std=torch.ones(self.rows, self.cols, self.dim) * 0.4).cuda()

		# Create grid index matrix
		np_x, np_y = np.meshgrid(range(self.rows), range(self.cols))
		x = torch.from_numpy(np_x)
		y = torch.from_numpy(np_y)
		self.grid = torch.stack((x,y)).cuda()

		# Compute grid radii just the one time
		self.grid_dists = pdist2(
			self.grid.float().view(2, -1), 
			self.grid.float().view(2, -1),
			0).cuda()


	def update(self, x, lr, sigma):
		''' x is N x 3
		'''
		# Compute update weights given the curren learning rate and sigma
		weights = lr * torch.exp(-self.grid_dists / (2 * sigma**2))

		# Determine closest units on the grid and the difference between data
		 # and units
		min_idx, diff = self._find_bmu(x)



		# print 1e5*weights[min_idx, :].view(-1, self.rows, self.cols)[30]
		# print weights[min_idx, :].view(-1, self.rows, self.cols).shape
		# print self._ind2sub(min_idx[2])
		# print weights[min_idx, :][0].view(self.rows, self.cols)
		# print weights[:, min_idx]
		# print diff


		# Compute the weighted content update
		# N x R*C * 3
		update = weights[min_idx, :].view(-1, self.rows*self.cols, 1) * diff

		# Aggregate over all the data samples
		update = update.sum(0)


		# print weights[min_idx, :].view(-1, self.rows*self.cols, 1)[2].view(self.rows,self.cols)
		# print torch.norm(diff[2],2,-1).view(self.rows,self.cols)
		# print torch.norm(update[2].view(self.rows, self.cols, -1),2,-1)

		# print torch.norm(update.permute(1,0,2)[0].view(self.rows, self.cols, -1),2,2)
		# print update.view(self.rows, self.cols, -1)
		# exit()

		# Update the contents of the grid
		self.contents += update.view(self.rows, self.cols, -1)

		# Return the average magnitude of the update
		return torch.norm(update, 2, -1).mean()

	def _find_bmu(self, x):
		''' x is N x 3
		'''
		N = x.shape[0]

		# Compute the Euclidean distances of the data
		diff = x.view(-1, 1, self.dim) - self.contents.view(1, -1, self.dim)
		dist = (diff ** 2).sum(-1).sqrt()
		# dist is N x R*C

		# Find the index of the best matching unit
		_, min_idx = dist.min(1)

		# Return indices
		return min_idx, diff

	
	def _ind2sub(self, idx):
		r, c = ind2sub(idx, self.cols)
		return r, c		

	def  _sub2ind(self, r, c):
		return sub2ind(r, c, self.cols)

	def init_viz(self):
		VIS.scatter(
			X=np.random.rand(2, 3),
			Y=np.array([1,2]),
			env=ENV,
			win='points',
			opts=dict(
				title='SOM',
				legend=['SOM Contents', 'Data'],
				markersize=4,
				markercolor=np.array([[0,0,255], [255,0,0]])))
		VIS.image(
			np.ones((1, 256,256)) * 255.,
			env=ENV,
			win='grid')



	# contents is K x 3
	# data is N x 3
	def update_viz(self, init_contents, contents, data):

		# Construct labels
		np_one = np.ones(contents.view(-1,self.dim).shape[0]).astype(int)
		np_two = 2*np.ones(data.shape[0]).astype(int)
		np_three = 3*np.ones(init_contents.view(-1,self.dim).shape[0]).astype(int)

		pts = np.row_stack((
			contents.view(-1, self.dim).numpy(), 
			data.numpy(), 
			init_contents.view(-1, self.dim).numpy()))
		labels = np.hstack((np_one, np_two, np_three))

		VIS.scatter(
			X=pts,
			Y=labels,
			env=ENV,
			win='points',
			opts=dict(
				title='SOM',
				legend=['SOM Contents', 'Data', 'Initial SOM Contents'],
				markersize=4,
				markercolor=np.array([[0, 0, 255], [255,0,0], [0,255,0]])))

		X = np.c_[contents[...,0].view(-1).numpy(),
			contents[...,1].view(-1).numpy(),
			np.zeros(contents[...,1].view(-1).shape) if self.dim==2 else \
			contents[...,2].view(-1).numpy()]

		views = []
		for i in xrange(self.dim):
			views.append(contents[...,i].view(-1).numpy())
		X = np.c_[views].transpose(1,0)
		print X.shape

		I = []
		J = []
		K = []
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				I.append(self._sub2ind(i, j))
				J.append(self._sub2ind(i,j+1))
				K.append(self._sub2ind(i+1,j+1))
				I.append(self._sub2ind(i, j))
				J.append(self._sub2ind(i+1,j))
				K.append(self._sub2ind(i+1,j+1))
		Y = np.c_[I, J, K]
		VIS.mesh(
			X=X,
			Y=Y,
			env=ENV,
			win='mesh',
			opts=dict(
				markersize=4,
				opacity=0.3))


		# Determine color map as a 3 x rows x cols image
		colors = contents.clone()
		mag = torch.norm(colors, 2, 1, True)
		colors /= mag
		colors = np.minimum(mag, 1.) * 255 * (colors + 1.) / 2.
		colors = colors.permute(2, 0, 1)

		
		colors = torch.zeros(1, contents.shape[0]-1, contents.shape[1]-1)
		for i in xrange(contents.shape[0]-1):
			for j in xrange(contents.shape[1]-1):
				colors[0, i, j] = ((contents[i,j,:] - contents[i,j+1,:]).norm() + 
					(contents[i,j,:] - contents[i+1,j,:]).norm() + 
					(contents[i,j,:] - contents[i+1,j+1,:]).norm()) / 3.

		colors /= colors.max()
		colors *= 255.

		# Upsample to 512 x 512
		colors = torch.nn.functional.upsample(torch.autograd.Variable(colors.unsqueeze(0)), scale_factor=100, mode='nearest').squeeze(0).numpy()


		VIS.image(
			colors,
			env=ENV,
			win='grid')



def main():

	# Create SOM
	dim = 2
	som = SOM(10,10,dim)
	som.initialize()

	init_contents = som.contents.clone()

	# Initialize visualization
	som.init_viz()

	# Initial SOM parameters
	lr = 0.2
	sigma = 0.5

	for i in xrange(1000000):
		# Generate some test data by sampling from a cube
		# data = (2 * torch.rand(100, 3) - 1)

		# Generate some test data by sampling from a spherical surface
		# data = torch.randn(50,3)
		# data /= torch.norm(data, 2, 1, True)

		data = generateSquareArea(100)

		# Put data on GPU
		data = data.cuda()

		# Update the SOM
		res = som.update(data, lr, sigma)

		# Decay the parameters
		if i % 500 == 0:
			lr *= 0.99
			sigma *= 0.99
			print 'Res: ', res
			print 'New LR: ', lr
			print 'New Sigma: ', sigma

		if i % 500 == 0:
			som.update_viz(
				init_contents.cpu(), 
				som.contents.cpu(), 
				data.cpu())



if __name__ == '__main__':
	main()
