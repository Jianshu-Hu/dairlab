import numpy as np
import matplotlib.pyplot as plt

# X = np.array([[0, 0], [1, 0.5], [1, 1.5], [0, 1], [0, 0]])

Z = np.array([0, 1, 1, 0, 0])
Z2 = np.array([0, 0.5, 1.5, 1, 0])
X = np.ones((Z.shape[0],2))
X[:,0] = Z
X[:,1] = Z2
t1 = plt.Polygon(X, color='red')
plt.gca().add_patch(t1)
plt.plot(Z, Z2, color='red')

plt.show()

