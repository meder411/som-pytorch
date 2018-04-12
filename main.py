import torch
import numpy as np

import visdom

from som import IterativeSOM
from som_utils import *


VIS = visdom.Visdom()
ENV = 'SOM'
SHAPE = 'square'


# Initial SOM parameters
ROWS = 10
COLS = 10
LR = 0.2
SIGMA = 0.5


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

	# Create SOM
	if SHAPE == 'circle':
		dim = 2
	elif SHAPE == 'sphere':
		dim = 2
	elif SHAPE == 'cube_vol':
		dim = 3
	elif SHAPE == 'square':
		dim = 2

	tmp = SOM()

	lr = LR
	sigma = SIGMA

	vis = Viz(VIS, ENV)
	som = IterativeSOM(ROWS, COLS, dim, vis)
	som.initialize()

	init_contents = som.contents.clone()

	# Initialize visualization
	som.init_viz()


	for i in xrange(100000):
		# Generate some test data
		if SHAPE == 'circle':
			data = generateCirclePerimeter(100)
		elif SHAPE == 'sphere':
			data = generateSphereSurface(100)
		elif SHAPE == 'cube_vol':
			data = generateCubeVolume(100)
		elif SHAPE == 'square':
			data = generateSquareArea(100)

		# Put data on GPU
		data = data.cuda()

		# Update the SOM
		res = som.update(data, lr, sigma)

		# Decay the parameters
		if i % 500 == 0:
			lr *= 0.9
			sigma *= 0.9
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
