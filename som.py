import torch
import numpy as np

import visdom

from abc import ABCMeta, abstractmethod
from som_utils import *

ENV = 'SOM'



class SOM(object):
	__metaclass__ = ABCMeta

	def __init__(self, rows=4, cols=4, dim=3, vis=None):
		self.rows = rows
		self.cols = cols
		self.dim = dim
		self.vis = vis
		self.contents = None
		self.grid = None
		self.grid_dists = None


	@abstractmethod
	def initialize(self):
		pass

	@abstractmethod
	def update(self):
		pass

	@abstractmethod
	def find_bmu(self):
		pass


	def _ind2sub(self, idx):
		r, c = ind2sub(idx, self.cols)
		return r, c		

	def  _sub2ind(self, r, c):
		return sub2ind(r, c, self.cols)




class IterativeSOM(SOM):

	def __init__(self, rows=4, cols=4, dim=3, vis=None):
		SOM.__init__(self, rows, cols, dim, vis)


	def initialize(self):
		# Randomly initialize unit contents
		# self.contents = torch.normal(
		# 	mean=0.0, 
		# 	std=torch.ones(self.rows, self.cols, self.dim) * 0.4).cuda()

		self.contents = grid(self.rows, self.cols, self.dim).cuda()

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
		min_idx, diff = self.find_bmu(x)

		# Compute the weighted content update
		# N x R*C * 3
		update = weights[min_idx, :].view(-1, self.rows*self.cols, 1) * diff

		# print torch.norm(update,2,-1).view(100, self.cols, self.rows)

		# Aggregate over all the data samples
		update = update.sum(0)

		# Update the contents of the grid
		self.contents += update.view(self.rows, self.cols, -1)

		# Return the average magnitude of the update
		return torch.norm(update, 2, -1).mean()

	def find_bmu(self, x):
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

	

	def init_viz(self):

		if self.vis is None:
			return

		self.vis.scatter(
			X=np.random.rand(2, 3),
			Y=np.array([1,2]),
			env=ENV,
			win='points',
			opts=dict(
				title='SOM',
				legend=['SOM Contents', 'Data'],
				markersize=4,
				markercolor=np.array([[0,0,255], [255,0,0]])))
		self.vis.image(
			np.ones((1, 256,256)) * 255.,
			env=ENV,
			win='grid')



	# contents is K x 3
	# data is N x 3
	def update_viz(self, init_contents, contents, data):

		if self.vis is None:
			return

		# Construct labels
		np_one = np.ones(contents.view(-1,self.dim).shape[0]).astype(int)
		np_two = 2*np.ones(data.shape[0]).astype(int)
		np_three = 3*np.ones(init_contents.view(-1,self.dim).shape[0]).astype(int)

		pts = np.row_stack((
			contents.view(-1, self.dim).numpy(), 
			data.numpy(), 
			init_contents.view(-1, self.dim).numpy()))
		labels = np.hstack((np_one, np_two, np_three))

		self.vis.scatter(
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


		I = []
		J = []
		K = []
		for i in xrange(self.rows-1):
			for j in xrange(self.cols-1):
				I.append(self._sub2ind(i, j))
				J.append(self._sub2ind(i,j+1))
				K.append(self._sub2ind(i+1,j+1))
				I.append(self._sub2ind(i, j))
				J.append(self._sub2ind(i+1,j))
				K.append(self._sub2ind(i+1,j+1))

		Y = np.c_[I, J, K]
		self.vis.mesh(
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


		self.vis.image(
			colors,
			env=ENV,
			win='grid')


