import numpy as np

def make_cumulative_prob_fun(probs):
	cumulatives = np.zeros(probs.shape)
	for j in range(cumulatives.shape[0]):
		cumulatives[j] = np.sum(probs[0:(j+1)])
	def find_index(val):
		assert 0 <= val
		assert val <= 1
		ind = 0
		while True:
			if cumulatives[ind] <= val:
				return ind
			else:
				ind += 1
	return find_index

def mixture_of_gaussians(npoints, probs, means, eigvals, angles):
	"""
	Sample points from a multimodal Guassian defined by
	NumPy arrays probs, means, and angles, all
	of identical shape (n, ), with eigvals having shape (2, n)
	"""
	num_gaussians = probs.shape[0]
	for input_array in [probs, means, angles]:
		assert input_array.shape == (num_gaussians, )
	assert eigvals.shape == (2, num_gaussians)
	assert np.isclose(np.sum(probs), 1)

	# Generate linear transformation matrices
	scal_mats = [np.diag(eigvals[:, j]) for j in range(eigvals.shape[1])]
	rot_mats = [np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle]]) for angle in angles]
	lin_mats = [rot_mat @ scal_mat for rot_mat, scal_mat in zip(rot_mats, scal_mats)]

	# Sample points from unit Gaussian in R^2
	unit_gauss_points = np.random.randn((npoints, 2))

	# Assign each row in unit_gauss_points to a Gaussian
	index_sampler = make_cumulative_prob_fun(probs)
	inds = [index_sampler(val) for val in np.random.rand(npoints)]

	# Map the unit Gaussian to Gaussian corresponding to each index
	points = []
	for j, ind in enumerate(inds):
		points.append(lin_mat[ind] @ unit_gauss_points[j, :] + means[ind])
	points = np.stack(points, axis=0)
	
	return points
