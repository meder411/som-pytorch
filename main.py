import torch

import visdom

from som import *
from som_utils import *

import time


# Visualization parameters
VIS = visdom.Visdom()
ENV = 'SOM'


# Initial SOM parameters
ROWS = 12
COLS = 12
LR = 0.2
SIGMA = 0.5
SHAPE = 'sphere'
N = 500


def generateSphereSurface(N):
	data = torch.randn(N, 3)
	data /= torch.norm(data, 2, 1, True)
	return data

def generateCubeVolume(N):
	return 2 * torch.rand(N, 3) - 1

def generateSquareArea(N):
	return 2 * torch.rand(N, 2) - 1

def generateCirclePerimeter(N):
	data = 2 * torch.randn(N, 2) - 1
	data /= torch.norm(data, 2, 1, True)
	return data


def parse_shape(shape):
	# Parse test data distribution
	if shape == 'circle':
		dim = 2
		init_shape = '2dgrid'
	elif shape == 'sphere':
		dim = 3
		init_shape = 'ball'
	elif shape == 'cube_vol':
		dim = 3
		init_shape = 'ball'
	elif shape == 'square':
		dim = 2
		init_shape = '2dgrid'

	return dim, init_shape


def iterative_main():

	# Parse test data distribution
	dim, init_shape = parse_shape(SHAPE)

	# Create SOM
	lr = LR
	sigma = SIGMA
	vis = Viz(VIS, ENV)
	som = BatchSOM(ROWS, COLS, dim, vis)
	som.initialize(init_shape)

	# Store the initial SOM contents for visualization purposes
	init_contents = som.contents.clone()

	for i in xrange(10000):
		# Generate some test data
		if SHAPE == 'circle':
			data = generateCirclePerimeter(N)
		elif SHAPE == 'sphere':
			data = generateSphereSurface(N)
		elif SHAPE == 'cube_vol':
			data = generateCubeVolume(N)
		elif SHAPE == 'square':
			data = generateSquareArea(N)

		# Put data on GPU
		data = data.cuda()

		# Update the SOM
		res = som.update(data, sigma, True)

		# Decay the parameters
		if i % 500 == 0:
			lr *= 0.9
			sigma *= 0.9
			print 'New LR: ', lr
			print 'New Sigma: ', sigma

		# Visualize the SOM
		if i % 5 == 0:
			som.update_viz(
				init_contents.cpu(), 
				som.contents.cpu(), 
				data.cpu())
			print 'Res: ', res


def batch_main():

	# Parse test data distribution
	dim, init_shape = parse_shape(SHAPE)

	# Create SOM
	lr = LR
	sigma = SIGMA
	vis = Viz(VIS, ENV)
	som = BatchSOM(ROWS, COLS, dim, vis)
	som.initialize(init_shape)

	# Store the initial SOM contents for visualization purposes
	init_contents = som.contents.clone()

	start = time.time()
	for i in xrange(10):
		# Generate some test data
		if SHAPE == 'circle':
			data = generateCirclePerimeter(N)
		elif SHAPE == 'sphere':
			data = generateSphereSurface(N)
		elif SHAPE == 'cube_vol':
			data = generateCubeVolume(N)
		elif SHAPE == 'square':
			data = generateSquareArea(N)

		# Put data on GPU
		data = data.cuda()

		# Update the SOM
		res = som.update(data, sigma, True)

		# # Visualize the SOM
		# if i % 5 == 0:
		# 	som.update_viz(
		# 		init_contents.cpu(), 
		# 		som.contents.cpu(), 
		# 		data.cpu())
		# 	print 'Res: ', res

		# time.sleep(2)
	print time.time() - start
	print 'Res: ', res
	som.update_viz(
		init_contents.cpu(), 
		som.contents[som.grid_used.nonzero(),:].cpu(), 
		data.cpu())




if __name__ == '__main__':
	batch_main()
