import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import csv
import os

# setting of program
robot_option = 1

# plot setting
task_space1 = 'uniform_grid'
task_space2 = 'random_sample'
task_space3 = 'cassie'

dir1 = '../dairlib_data/'+task_space3+'/robot_' + str(robot_option) + \
'_iter200/'
label6 = 'restricted number of sample using new initial guess 2 '
line_type6 = 'k--'

dir2 = '../dairlib_data/'+task_space3+'/robot_' + str(robot_option) + \
'_iter70/'
label7 = 'uniform grid with original initial guess'
line_type7 = 'k-'


def average_cost_several_iter(iter_start, iter_end, n, dir, line_type, label_name, normalized, dir_nominal):
    # get nominal cost from iteration 0
    if normalized == 1:
        nominal_cost = []
        j = 0
        while os.path.isfile(dir_nominal+str(0)+'_'+str(j)+'_c.csv'):
            if np.genfromtxt(dir_nominal+str(0)+'_'+str(j)+'_is_success.csv', delimiter=","):
                nominal_cost.append(np.genfromtxt(dir_nominal+str(0)+'_'+str(j)+'_c.csv', delimiter=","))
                j = j+1
        nominal_cost = np.array(nominal_cost).sum()/len(nominal_cost)
        print("normalize the cost by nominal cost:", nominal_cost)
    else:
        nominal_cost = 1
        print("without normalizing the cost by nominal cost")

    aver_cost = []
    for i in range(iter_start, iter_end+1):
        solve_time = []
        j = 0
        while os.path.isfile(dir+str(i)+'_'+str(j)+'_c.csv'):
            if np.genfromtxt(dir+str(i)+'_'+str(j)+'_is_success.csv', delimiter=","):
                solve_time.append(np.genfromtxt(dir+str(i)+'_'+str(j)+'_c.csv', delimiter=","))
                j = j+1
        aver_cost.append(np.array(solve_time).sum()/len(solve_time))

    # average the cost every n iterations
    cost = np.array(aver_cost)
    aver_cost = []
    j = 0
    while j < len(cost):
        if len(cost)-j < n:
            aver_cost.append(cost[j:len(cost)].sum()/(len(cost)-j))
        else:
            aver_cost.append(cost[j:j+n].sum()/n)
        j = j + n
    ax1.plot(range(1, len(aver_cost)+1), aver_cost, line_type, linewidth=3.0, label=label_name)


fig1 = plt.figure(num=1, figsize=(6.4, 4.8))
ax1 = fig1.gca()
average_cost_several_iter(1, 200, 25, dir1, line_type6, label6, 1, dir2)
average_cost_several_iter(1, 70, 4, dir2, line_type7, label7, 1, dir2)
plt.xlabel('Every 100 samples')
plt.ylabel('Average cost of samples')
plt.legend()
plt.show()