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
		self.grid_used = None
		self.grid_dists = None



	# Abstract methods for all SOMs
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
	def initialize(self, shape):

		if shape == '2dgrid':
			self.contents = grid(self.rows, self.cols, self.dim).cuda()
		elif shape == 'ball':
			self.contents = torch.normal(
				0, 0.2*torch.ones(self.rows, self.cols, self.dim)).cuda()

		# Create grid index matrix
		np_x, np_y = np.meshgrid(range(self.rows), range(self.cols))
		x = torch.from_numpy(np_x)
		y = torch.from_numpy(np_y)
		self.grid = torch.stack((x,y)).cuda()
		self.grid_used = torch.zeros(self.rows, self.cols).long().cuda()

		# Compute grid radii just the one time
		self.grid_dists = pdist2(
			self.grid.float().view(2, -1), 
			self.grid.float().view(2, -1),
			0).cuda()


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

# =========================================================================== #
# =========================================================================== #

class ParallelBatchSOM(SOM):

	def __init__(self, rows=4, cols=4, dim=3, batches=1, vis=None):
		SOM.__init__(self, rows, cols, dim, vis)
		self.batches = batches


	def initialize(self, shape):
		# Initialize the standard BatchSOM
		SOM.initialize(self, shape)

		# Replicate the initialize BatchSOM B times
		self.contents = self.contents.repeat(self.batches, 1, 1, 1)
		self.grid = self.grid.repeat(self.batches, 1, 1, 1)
		self.grid_used = self.grid_used.repeat(self.batches, 1, 1)


	def update(self, x, sigma, weighted=False):
		''' x is N x 3'''
		# Compute update weights given the current learning rate and sigma
		weights = self.compute_weights(sigma, weighted)

		# Determine closest units on the grid and the difference between data
		# and units
		min_idx = self.find_bmu(x)
		
		# Adjust indices for batching
		batch_num = torch.LongTensor(range(self.batches)).cuda()
		min_idx = batch_num.view(-1,1) * self.rows*self.cols + min_idx

		# Compute the frequency with which each node is the BMU
		freq_data = torch.zeros(self.batches, self.rows*self.cols).cuda()
		freq_data.view(-1).index_add_(0, min_idx.view(-1), torch.ones(x.shape[:2]).cuda().view(-1))
		
		# Store the update frequency for each node
		self.grid_used += (freq_data != 0).view(self.batches, self.rows, self.cols).long()

		# Compute aggregate data values for each neighborhood
		sum_data = torch.zeros(self.batches, self.rows*self.cols, self.dim).cuda()

		sum_data.view(-1, self.dim).index_add_(0, min_idx.view(-1), x.view(-1, self.dim))
		avg_data = sum_data / freq_data.unsqueeze(-1)


		# Weight the neighborhood impacts by the frequency data
		freq_weights = weights * freq_data.unsqueeze(-1)
		print freq_weights
		exit()
		
		# Use the existing node contents for any nodes with no nearby data
		unused_idx = (freq_data == 0).nonzero()
		if unused_idx.shape:
			avg_data[unused_idx, :] = self.contents.view(-1, self.dim)[unused_idx, :]
		
		# Compute the update
		update_num = (freq_weights.unsqueeze(2) * avg_data).sum(1)
		update_denom = freq_weights.sum(1)
		update = update_num / update_denom.unsqueeze(1)

		# Determine which nodes are actually update-able
		update_idx = update_denom.nonzero()

		# Copy the old node contents for later update magnitude computation
		old_contents = self.contents.clone()

		# Update the nodes
		self.contents.view(-1, self.dim)[update_idx, :] = update[update_idx, :]

		# Return the average magnitude of the update
		return torch.norm(self.contents-old_contents, 2, -1).mean()


	def compute_weights(self, sigma, weighted=False):
		if weighted:
			return torch.exp(-self.grid_dists / (2 * sigma**2))
		else:
			return (self.grid_dists < sigma).float()


	def find_bmu(self, x, k=1):
		''' x is B x N x 3'''
	
		# Batch size and point set size
		B, N, _ = x.shape

		# Compute the Euclidean distances of the data
		diff = x.unsqueeze(2) - self.contents.view(B, 1, -1, self.dim)
		dist = (diff ** 2).sum(-1).sqrt()

		# Find the index of the best matching unit
		_, min_idx = dist.topk(k=k, dim=-1, largest=False)

		# Return indices
		return min_idx[...,0].squeeze()


	# contents is K x 3
	# data is N x 3
	def update_viz(self, init_contents, contents, data):

		print init_contents.shape
		print contents.shape
		print data.shape

		if self.vis is None:
			return

		B = data.shape[0]

		for i in xrange(B):

			# Construct labels
			np_one = np.ones(contents[i].view(-1,self.dim).shape[0]).astype(int)
			np_two = 2*np.ones(data[i].shape[0]).astype(int)
			np_three = 3*np.ones(init_contents[i].view(-1,self.dim).shape[0]).astype(int)

			# Stack the points into the right format for visdom
			pts = np.row_stack((
				contents[i].view(-1, self.dim).numpy(), 
				data[i].numpy(), 
				init_contents[i].view(-1, self.dim).numpy()))
			labels = np.hstack((np_one, np_two, np_three))

			print pts.shape
			print labels.shape

			# Plot the data in a scatter plot
			self.vis.visdom.scatter(
				X=pts,
				Y=labels,
				env=self.vis.env,
				win='points' + str(i),
				opts=dict(
					title='SOM',
					legend=['SOM Contents', 'Data', 'Initial SOM Contents'],
					markersize=4,
					markercolor=np.array([[0, 0, 255], [255,0,0], [0,255,0]])))

			# Prep the data for mesh visualization of the SOM nodes
			X = np.c_[contents[i,...,0].view(-1).numpy(),
				contents[i,...,1].view(-1).numpy(),
				np.zeros(contents[i,...,1].view(-1).shape) if self.dim==2 else \
				contents[i,...,2].view(-1).numpy()]
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
				win='mesh' + str(i),
				opts=dict(
					markersize=4,
					opacity=0.3))


# =========================================================================== #
# =========================================================================== #


class BatchSOM(SOM):

	def __init__(self, rows=4, cols=4, dim=3, vis=None):
		SOM.__init__(self, rows, cols, dim, vis)


	def update(self, x, sigma, weighted=False):
		''' x is N x 3'''
		# Compute update weights given the current learning rate and sigma
		weights = self.compute_weights(sigma, weighted)

		# Determine closest units on the grid and the difference between data
		 # and units
		min_idx = self.find_bmu(x)

		# Compute the frequency with which each node is the BMU
		freq_data = torch.zeros(self.rows*self.cols).cuda()
		freq_data.index_add_(0, min_idx, torch.ones(x.shape[0]).cuda())

		# Store the update frequency for each node
		self.grid_used += (freq_data != 0).view(self.rows, self.cols).long()

		# Compute aggregate data values for each neighborhood
		sum_data = torch.zeros(self.rows*self.cols, self.dim).cuda()
		sum_data.index_add_(0, min_idx, x)
		avg_data = sum_data / freq_data.view(-1,1)

		# Weight the neighborhood impacts by the frequency data
		freq_weights = weights * freq_data.view(-1, 1)
		
		# Use the existing node contents for any nodes with no nearby data
		unused_idx = (freq_data == 0).nonzero()
		if unused_idx.shape:
			avg_data[unused_idx, :] = self.contents.view(-1, self.dim)[unused_idx, :]
		
		# Compute the update
		update_num = (freq_weights.unsqueeze(2) * avg_data).sum(1)
		update_denom = freq_weights.sum(1)
		update = update_num / update_denom.unsqueeze(1)

		# Determine which nodes are actually update-able
		update_idx = update_denom.nonzero()

		# Copy the old node contents for later update magnitude computation
		old_contents = self.contents.clone()

		# Update the nodes
		self.contents.view(-1, self.dim)[update_idx, :] = update[update_idx, :]

		# Return the average magnitude of the update
		return torch.norm(self.contents-old_contents, 2, -1).mean()


	def compute_weights(self, sigma, weighted=False):
		if weighted:
			return torch.exp(-self.grid_dists / (2 * sigma**2))
		else:
			return (self.grid_dists < sigma).float()


	def find_bmu(self, x, k=1):
		''' x is N x 3'''
		N = x.shape[0]

		# Compute the Euclidean distances of the data
		diff = x.view(-1, 1, self.dim) - self.contents.view(1, -1, self.dim)
		dist = (diff ** 2).sum(-1).sqrt()
		# dist is N x R*C

		# Find the index of the best matching unit
		_, min_idx = dist.topk(k=k, dim=1, largest=False)

		# Return indices
		return min_idx.squeeze()



# =========================================================================== #
# =========================================================================== #


class IterativeSOM(SOM):

	def __init__(self, rows=4, cols=4, dim=3, vis=None):
		SOM.__init__(self, rows, cols, dim, vis)


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
