import torch
import numpy as np

import visdom

from abc import ABCMeta, abstractmethod
from som_utils import *




class SOM(object):
	__metaclass__ = ABCMeta

	# Constructor
	def __init__(self, rows=4, cols=4, dim=3, vis=None):
		self.rows = rows
		self.cols = cols
		self.dim = dim
		self.vis = vis
		self.contents = None
		self.grid = None
		self.grid_dists = None


	# Abstract methods for all SOMs
	@abstractmethod
	def initialize(self):
		pass

	@abstractmethod
	def update(self):
		pass

	@abstractmethod
	def compute_weights(self, sigma, weighted=False):
		pass

	@abstractmethod
	def find_bmu(self):
		pass


	# Base class methods that (should) have use for all types of SOMs
	def _ind2sub(self, idx):
		r, c = ind2sub(idx, self.cols)
		return r, c		


	def  _sub2ind(self, r, c):
		return sub2ind(r, c, self.cols)


	# contents is K x 3
	# data is N x 3
	def update_viz(self, init_contents, contents, data):

		if self.vis is None:
			return

		# Construct labels
		np_one = np.ones(contents.view(-1,self.dim).shape[0]).astype(int)
		np_two = 2*np.ones(data.shape[0]).astype(int)
		np_three = 3*np.ones(init_contents.view(-1,self.dim).shape[0]).astype(int)

		# Stack the points into the right format for visdom
		pts = np.row_stack((
			contents.view(-1, self.dim).numpy(), 
			data.numpy(), 
			init_contents.view(-1, self.dim).numpy()))
		labels = np.hstack((np_one, np_two, np_three))

		# Plot the data in a scatter plot
		self.vis.visdom.scatter(
			X=pts,
			Y=labels,
			env=self.vis.env,
			win='points',
			opts=dict(
				title='SOM',
				legend=['SOM Contents', 'Data', 'Initial SOM Contents'],
				markersize=4,
				markercolor=np.array([[0, 0, 255], [255,0,0], [0,255,0]])))

		# Prep the data for mesh visualization of the SOM nodes
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

		# Visualize the SOM nodes as a mesh
		self.vis.visdom.mesh(
			X=X,
			Y=Y,
			env=self.vis.env,
			win='mesh',
			opts=dict(
				markersize=4,
				opacity=0.3))



# =========================================================================== #
# =========================================================================== #



class BatchSOM(SOM):

	def __init__(self, rows=4, cols=4, dim=3, vis=None):
		SOM.__init__(self, rows, cols, dim, vis)


	def initialize(self):
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


	def update(self, x, sigma, weighted=False):
		''' x is N x 3'''
		# Compute update weights given the curren learning rate and sigma
		weights = self.compute_weights(sigma, weighted)

		# Determine closest units on the grid and the difference between data
		 # and units
		min_idx = self.find_bmu(x)


		# Compute aggregate data values for each neighborhood
		sum_data = torch.zeros(self.rows*self.cols, self.dim).cuda()
		sum_data.index_add_(0, min_idx, x)

		# Compute the frequency with which each node is the BMU
		freq_data = torch.zeros(self.rows*self.cols).cuda()
		freq_data.index_add_(0, min_idx, torch.ones(x.shape[0]).cuda())

		# Weight the neighborhood impacts by the frequency data
		freq_weights = weights * freq_data.view(-1, 1)
		print 'freq_weights'
		print freq_weights[min_idx, :].view(-1, self.rows, self.cols)
		print freq_weights

		# Compute the update
		update_num = (freq_weights[min_idx, :].unsqueeze(2) * sum_data).sum(0)
		update_denom = freq_weights.sum(0)

		# print 'update_num'
		# print update_num
		# print 'update_denom'
		# print update_denom

		# update = update_num / update_denom.unsqueeze(1)

		# print 'update'
		# print update

		# # Determine which nodes are actually update-able
		# update_idx = update_denom.nonzero()

		# print 'contents before'
		# print self.contents
		# self.contents.view(-1, self.dim)[update_idx, :] = update[update_idx, :]
		# print 'contents after'
		# print self.contents


		exit()

		# Compute the weighted content update
		# N x R*C * 3
		update = weights[min_idx, :].view(-1, self.rows*self.cols, 1) * diff

		# Aggregate over all the data samples
		update = update.sum(0)

		# Update the contents of the grid
		self.contents += update.view(self.rows, self.cols, -1)

		# Return the average magnitude of the update
		return torch.norm(update, 2, -1).mean()


	def compute_weights(self, sigma, weighted=False):
		if weighted:
			return torch.exp(-self.grid_dists / (2 * sigma**2))
		else:
			return (self.grid_dists < sigma).float()


	def find_bmu(self, x):
		''' x is N x 3'''
		N = x.shape[0]

		# Compute the Euclidean distances of the data
		diff = x.view(-1, 1, self.dim) - self.contents.view(1, -1, self.dim)
		dist = (diff ** 2).sum(-1).sqrt()
		# dist is N x R*C

		# Find the index of the best matching unit
		_, min_idx = dist.min(1)

		# Return indices
		return min_idx



# =========================================================================== #
# =========================================================================== #


class IterativeSOM(SOM):

	def __init__(self, rows=4, cols=4, dim=3, vis=None):
		SOM.__init__(self, rows, cols, dim, vis)


	def initialize(self):
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


	def update(self, x, lr, sigma, weighted=False):
		''' x is N x 3'''

		# Compute update weights given the curren learning rate and sigma
		weights = lr * self.compute_weights(sigma, weighted)

		# Determine closest units on the grid and the difference between data
		 # and units
		min_idx, diff = self.find_bmu(x)

		# Compute the weighted content update
		# N x R*C * 3
		update = weights[min_idx, :].view(-1, self.rows*self.cols, 1) * diff

		# Aggregate over all the data samples
		update = update.sum(0)

		# Update the contents of the grid
		self.contents += update.view(self.rows, self.cols, -1)

		# Return the average magnitude of the update
		return torch.norm(update, 2, -1).mean()


	def compute_weights(self, sigma, weighted=False):
		if weighted:
			return torch.exp(-self.grid_dists / (2 * sigma**2))
		else:
			return (self.grid_dists < sigma).float()


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
