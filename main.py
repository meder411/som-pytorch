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
		# print "lr"
		# print lr
		# print "sigma"
		# print sigma
		weights = lr * torch.exp(-self.grid_dists / (2 * sigma**2))

		# Determine closest units on the grid and the difference between data
		 # and units
		min_idx, diff = self._find_bmu(x)

		# Compute the weighted content update
		update = (weights[:, min_idx].unsqueeze(2) * diff).sum(1)

		# print "weights"
		# print weights
		# print "diff"
		# print diff
		# print "update"
		# print update
		# print "max_update"
		# print update.max()

		# Update the contents of the grid
		self.contents += update.view(self.rows, self.cols, -1)

		# Return the average magnitude of the update
		return torch.norm(update, 2, 1).mean()

	def _find_bmu(self, x):
		''' x is N x 3
		'''
		N = x.shape[0]

		# Compute the Euclidean distances of the data
		diff = x - self.contents.view(-1, 1, self.dim).expand(-1, N, -1)
		dist = (diff ** 2).sum(-1).sqrt()

		# Find the index of the best matching unit
		_, min_idx = dist.min(0)

		# Return indices
		return min_idx, diff

	
	def _ind2sub(self, idx):
		r, c = ind2sub(idx, self.cols)
		return r, c		


def init_viz():
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


# contents is K x 3
# data is N x 3
def update_viz(init_contents, contents, data):

	# Construct labels
	np_one = np.ones(contents.shape[0]).astype(int)
	np_two = 2*np.ones(data.shape[0]).astype(int)
	np_three = 3*np.ones(init_contents.shape[0]).astype(int)

	pts = np.row_stack((contents.numpy(), data.numpy(), init_contents.numpy()))
	labels = np.hstack((np_one, np_two, np_three))

	VIS.scatter(
		X=pts,
		Y=labels,
		env=ENV,
		win='points',
		opts=dict(
			title='SOM',
			legend=['SOM Contents', 'Data'],
			markersize=4,
			markercolor=np.array([[0, 0, 255], [255,0,0], [0,255,0]])))



def main():

	# Create SOM
	som = SOM()
	som.initialize()

	init_contents = som.contents

	# Initialize visualization
	init_viz()

	# Initial SOM parameters
	lr = 0.2
	sigma = 0.2

	for i in xrange(10000):
		# Generate some test data by sampling from a cube
		data = (2 * torch.rand(100,3) - 1).cuda()


		# Update the SOM
		res = som.update(data, lr, sigma)
		print 'Res: ', res

		# Decay the parameters
		if i % 50 == 0:
			lr *= 0.99
			sigma *= 0.99
			update_viz(init_contents.cpu(), som.contents.view(-1,3).cpu(), data.cpu())
			print 'New LR: ', lr
			print 'New Sigma: ', sigma


		# time.sleep(0.25)




if __name__ == '__main__':
	main()
