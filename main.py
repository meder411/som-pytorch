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


def main():

	# Parse test data distribution
	if SHAPE == 'circle':
		dim = 2
	elif SHAPE == 'sphere':
		dim = 3
	elif SHAPE == 'cube_vol':
		dim = 3
	elif SHAPE == 'square':
		dim = 2

	# Create SOM
	lr = LR
	sigma = SIGMA
	vis = Viz(VIS, ENV)
	som = BatchSOM(ROWS, COLS, dim, vis)
	som.initialize()

	# Store the initial SOM contents for visualization purposes
	init_contents = som.contents.clone()

	for i in xrange(50):
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
		# res = som.update(data, 2, False)

		# Decay the parameters
		if i % 500 == 0:
			lr *= 0.9
			sigma *= 0.9
			print 'Res: ', res
			print 'New LR: ', lr
			print 'New Sigma: ', sigma

		# Visualize the SOM
		if i % 1 == 0:
			som.update_viz(
				init_contents.cpu(), 
				som.contents.cpu(), 
				data.cpu())

		time.sleep(2)


if __name__ == '__main__':
	main()
