import torch
import numpy as np
from torch import tensor
import matplotlib.pyplot as plt

robot_option = 1
if robot_option == 0:
    robot = 'five_link/'
else:
    if robot_option == 1:
        robot = 'cassie/'

# plot setting
task_space1 = '3d'
dir1 = '../dairlib_data/'+robot+task_space1+'/robot_' + str(robot_option) + \
'_3d_3_iter200_revised_code/'

num_list = []

for iteration in range(2, 200):
    theta_s = tensor(np.genfromtxt(dir1 + str(iteration) + '_theta_s.csv', delimiter=","))
    theta_sDDot = tensor(np.genfromtxt(dir1 + str(iteration) + '_theta_sDDot.csv', delimiter=","))
    theta = torch.cat((theta_s, theta_sDDot), 0)
    num = 0
    for j in range(1, iteration):
        theta_s_old = tensor(np.genfromtxt(dir1 + str(j) + '_theta_s.csv', delimiter=","))
        theta_sDDot_old = tensor(np.genfromtxt(dir1 + str(j) + '_theta_sDDot.csv', delimiter=","))
        theta_old = torch.cat((theta_s_old, theta_sDDot_old), 0)
        theta_dif = torch.norm(theta_old-theta)/torch.norm(theta)
        if theta_dif < 0.004:
            num = num+1
    num_list.append(num)

label1 = 'Num'
line_type1 = 'k-'
fig1 = plt.figure(num=1, figsize=(6.4, 4.8))
ax1 = fig1.gca()
ax1.plot(range(1, len(num_list)+1), num_list, line_type1, linewidth=3.0, label=label1)
plt.xlabel('Iteration')
plt.ylabel('Num of past iterations to use')
plt.legend()
plt.show()