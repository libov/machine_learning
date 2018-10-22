from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib import pyplot as plt

def plot_decision_regions(X, y, classifier, resolution = 0.02):
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	
	x1min, x1max = X[:,0].min() - 1, X[:,0].max()+1
	x2min, x2max = X[:,1].min() - 1, X[:,1].max()+1
	
	xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, resolution), np.arange(x2min, x2max, resolution))

	x = np.array([xx1.ravel(), xx2.ravel()])
	xt = x.T
	Z = classifier.predict(xt)
	Z = Z.reshape(xx1.shape)

	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	
	for idx, c1 in enumerate(np.unique(y)):
		plt.scatter(x=X[y==c1, 0], y=X[y==c1, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=c1)
