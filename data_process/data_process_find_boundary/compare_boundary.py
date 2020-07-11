"""
This function is used for comparing two cost landscape and plot the landscape with discrete color map.
Considering we search along several directions, we process the data along those directions
1.
For each direction, we compare the cost at each point on the searching line both in cost landscape 1 (C1) and
cost landscape. Set the value of the point according the cost.
2.
For those points which are not in two landscape, if it exists in C1, we set the value of this point -0.5,
else we set the value of this point 2
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

robot_option = 1
file_dir = '/Users/jason-hu/'
if robot_option == 1:
    robot = 'cassie/'
    dir1 = file_dir+'dairlib_data/find_boundary/' + robot + '2D_rom/4D_task_space/' + 'robot_' + str(robot_option) + \
           '_grid_iter50_sl_tr/'
    dir2 = file_dir+'dairlib_data/find_boundary/' + robot + '2D_rom/4D_task_space/' + 'robot_' + str(robot_option) + \
           '_nominal_sl_tr/'

# number of searching directions
n_direction = 16


# Note:decide which column of the task to plot according to the task dimensions
# Eg. column index 0 corresponds to stride length
task_1_idx = 0
task_2_idx = 3


def process_data_from_direction(dir1, dir_nominal):
    # need to add central point on the points list
    task0 = np.genfromtxt(dir1 + str(0) + '_' + str(0) + '_task.csv', delimiter=",")
    x0 = [task0[task_1_idx]]
    y0 = [task0[task_2_idx]]
    cost1 = np.genfromtxt(dir1 + str(0) + '_' + str(0) + '_c.csv', delimiter=",")
    cost2 = np.genfromtxt(dir_nominal + str(0) + '_' + str(0) + '_c.csv', delimiter=",")
    if cost1 > cost2:
        z0 = [1.5]
    else:
        z0 = [0.5]
    for i in range(n_direction):
        data_dir1 = np.genfromtxt(dir1 + str(i+1) + '_cost_list.csv', delimiter=",")
        data_dir2 = np.genfromtxt(dir_nominal + str(i+1) + '_cost_list.csv', delimiter=",")

        if data_dir1.shape[0] >= data_dir2.shape[0]:
            num_small = data_dir2.shape[0]
            num_large = data_dir1.shape[0]
        else:
            num_small = data_dir1.shape[0]
            num_large = data_dir2.shape[0]

        # process the points on the line
        x = []
        y = []
        z = []

        # set the value for intersected parts
        for j in range(num_small):
            cost1 = data_dir1[j, 1]
            cost2 = data_dir2[j, 1]
            # only consider reasonable point
            if (cost1 < 35) & (cost2 < 35):
                if cost1 > cost2:
                    z.append(1.5)
                else:
                    z.append(0.5)
                task = np.genfromtxt(dir1 + str(int(data_dir1[j, 0])) + '_' + str(0) + '_task.csv', delimiter=",")
                x.append(task[task_1_idx])
                y.append(task[task_2_idx])
        for j in range(num_small, num_large):
            if data_dir1.shape[0] >= data_dir2.shape[0]:
                # extended range
                task = np.genfromtxt(dir1 + str(int(data_dir1[j, 0])) + '_' + str(0) + '_task.csv', delimiter=",")
                x.append(task[task_1_idx])
                y.append(task[task_2_idx])
                z.append(-0.5)
            else:
                # shrunk range
                task = np.genfromtxt(dir_nominal + str(int(data_dir2[j, 0])) + '_' + str(0) + '_task.csv', delimiter=",")
                x.append(task[task_1_idx])
                y.append(task[task_2_idx])
                z.append(2)
        if len(x) > 10:
            x0 = x0 + x
            y0 = y0 + y
            z0 = z0 + z
    return np.array(x0), np.array(y0), np.array(z0)


fig, ax = plt.subplots()
x, y, z = process_data_from_direction(dir1, dir2)
# discrete color map
levels = [0, 1, 2]
colors = ['green', 'blue']
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
cmap.set_over('yellow')
cmap.set_under('red')
surf = ax.tricontourf(x, y, z, cmap=cmap, norm=norm, levels=levels, extend='both')
cbar = fig.colorbar(surf, shrink=0.5, aspect=6, extend='both')
cbar.ax.set_yticklabels(['0', '1', 'Infinity'])

ax.set_xlabel('Stride length')
ax.set_ylabel('Ground incline')
ax.set_title('Compare two cost landscapes')

plt.show()