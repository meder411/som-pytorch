import torch

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


def grid(r, c, dim):
	grid = torch.zeros(r, c, dim)
	for i in xrange(r):
		for j in xrange(c):
			if dim == 3:
				grid[i, j, :] = torch.FloatTensor([i, j, 0])
			else:
				grid[i, j, :] = torch.FloatTensor([i, j])

	grid[:,:,0] /= r
	grid[:,:,1] /= c
	grid[:,:,:2] = 2 * grid[:,:,:2] - 1
	
	return grid



class Viz(object):
	def __init__(self, visdom, env):
		self.visdom = visdom
		self.env = env