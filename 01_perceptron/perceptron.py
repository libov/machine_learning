import numpy as np
from matplotlib import pyplot as plt

class Perceptron(object):

	def __init__(self, eta=0.01, n_iter=10, verbose=False):

		self.eta = eta
		self.n_iter = n_iter
		self.verbose = verbose
		self.image_counter = 0

	def fit(self, X, Y):

		self.w_ = np.zeros(1+X.shape[1])
		#self.w_ = np.random.rand(1+X.shape[1])
		self.errors_ = []

		for _ in range(self.n_iter):
			if self.verbose:
				print('iteration=', _)
			errors = 0
			for xi, yi in zip(X, Y):
				if self.verbose:
					print('xi={0}, yi={1}; w={2} y={3}'.format(xi,yi,self.w_, self.predict(xi)))
				update = self.eta * (yi - self.predict(xi))
				self.w_[0] += update
				self.w_[1:] += update * xi
				errors += int(update != 0.0)
				if self.verbose and update !=0:
					print('new weights:', self.w_)
					print('errors=', errors)
					self.showCurrentModel(X)
			self.errors_.append(errors)
			if self.verbose:
				print('Iteration done, self.errors_=', self.errors_)
		return self
	
	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]
	
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)
	
	def showCurrentModel(self, X):
		x1 = X[:,0]
		x2 = X[:,1]
		plt.figure()
		plt.plot(x1,x2,'o')
		y_model=-self.w_[0]/self.w_[2] - x1*self.w_[1]/self.w_[2]
		plt.plot(x1,y_model)
		plt.savefig('{0}.png'.format(self.image_counter))
		self.image_counter+=1
		plt.close()
