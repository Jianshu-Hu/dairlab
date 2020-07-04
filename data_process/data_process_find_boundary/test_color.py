import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits import mplot3d

x = 5 * np.random.random(10)
y = 5 * np.random.random(10)
z = 5 * np.random.random(10)

levels = [0, 1, 2, 3, 4, 5]
colors = ['red', 'brown', 'yellow', 'green', 'blue']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
cmap.set_over('orange')
cmap.set_under('orange')

fig, ax = plt.subplots()
surf = ax.tricontourf(x, y, z, cmap=cmap, norm=norm, extend='both', levels=levels)
cbar = fig.colorbar(surf, shrink=0.5, aspect=6, extend='both')

plt.show()

# fig = plt.figure()
# ax = plt.axes(projection="3d")
#
# z_line = np.array([1, 1, 1, 1, 1])
# x_line = np.array([0, 0, 1, 1, 0])
# y_line = np.array([0, 1, 1, 0, 0])
# ax.plot3D(x_line, y_line, z_line, 'black', linewidth=2)
#
# z1_line = np.array([1, 1, 1, 1, 1])
# x1_line = np.array([0, 0, 0.5, 0.5, 0])
# y1_line = np.array([0, 0.5, 0.5, 0, 0])
# ax.plot3D(x1_line, y1_line, z1_line, 'black')

# ax.view_init(azim=0, elev=90)
# plt.show()
