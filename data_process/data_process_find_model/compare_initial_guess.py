import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import csv
import os
import time
import sys

# setting of program
iter_start = 2
iter_end = 500
robot_option = 1

if robot_option == 0:
    robot = 'five_link/'
else:
    if robot_option == 1:
        robot = 'cassie/'

# plot setting
task_space1 = 'uniform_grid'
task_space2 = 'random_sample'
task_space3 = '2d'
task_space4 = '3d'

dir1 = '../dairlib_data/'+robot+task_space4+'/robot_' + str(robot_option) + \
'_original_iter50/'
label1 = 'uniform grid with original initial guess'
line_type1 = 'k-'

dir2 = '../dairlib_data/'+robot+task_space4+'/robot_' + str(robot_option) + \
'_3d_3_iter150_new_method_test/'
label2 = 'using third power of norm'
line_type2 = 'k--'

dir3 = '../dairlib_data/'+robot+task_space4+'/robot_' + str(robot_option) + \
'_3d_3_iter200_revised_code/'
label3 = 'using third power of absolute value of diff'
line_type3 = 'k:'

print("check the dir name and plot setting carefully")


def distance(is_new_initial_guess, dir, line_type, label_name, n):
    diff = []

    for i in range(iter_start, iter_end+1):
        j = 0
        while os.path.isfile(dir+str(i)+'_'+str(j)+'_w.csv'):
            if is_new_initial_guess:
                initial_guess = np.genfromtxt(dir+str(i)+'_'+str(j)+'_initial_guess.csv', delimiter=",")
            else:
                initial_guess = np.genfromtxt(dir+str(i-1)+'_'+str(j)+'_w.csv', delimiter=",")
            result = np.genfromtxt(dir + str(i) + '_' + str(j) + '_w.csv', delimiter=",")
            diff.append(np.linalg.norm(result-initial_guess))
            j = j+1
    # average the distance every n samples
    diff = np.array(diff)
    aver_diff = []
    j = 0
    while j < len(diff):
        if len(diff)-j < n:
            aver_diff.append(diff[j:len(diff)].sum()/(len(diff)-j))
        else:
            aver_diff.append(diff[j:j+n].sum()/n)
        j = j + n

    ax1.plot(range(1, len(aver_diff)+1), aver_diff, line_type, linewidth=3.0, label=label_name)


def average_time(dir, line_type, label_name, n):
    solve_time = []
    for i in range(iter_start, iter_end+1):
        j = 0
        while os.path.isfile(dir+str(i)+'_'+str(j)+'_solve_time.csv'):
            solve_time.append(np.genfromtxt(dir+str(i)+'_'+str(j)+'_solve_time.csv', delimiter=","))
            j = j+1

    # average the distance every n samples
    solve_time = np.array(solve_time)
    aver_solve_time = []
    j = 0
    while j < len(solve_time):
        if len(solve_time) - j < n:
            aver_solve_time.append(solve_time[j:len(solve_time)].sum() / (len(solve_time) - j))
        else:
            aver_solve_time.append(solve_time[j:j + n].sum() / n)
        j = j + n

    ax1.plot(range(1, len(aver_solve_time) + 1), aver_solve_time, line_type, linewidth=3.0, label=label_name)

def plot_distance():
    # plot distancn
    n = 125
    distance(False, dir1, line_type1, label1, n)
    distance(True, dir2, line_type2, label2, n)
    distance(True, dir3, line_type3, label3, n)
    plt.xlabel('Every'+str(n)+'samples')
    plt.ylabel('Distance between initial guess and result')
    plt.legend()
    plt.show()

def plot_average_time():
    # plot average solve time
    n = 100
    average_time(dir1, line_type1, label1, n)
    average_time(dir2, line_type2, label2, n)
    average_time(dir3, line_type3, label3, n)
    plt.xlabel('Every '+str(n)+' samples')
    plt.ylabel('Average solve time for trajectory optimizations')
    plt.legend()
    plt.show()


fig1 = plt.figure(num=1, figsize=(6.4, 4.8))
ax1 = fig1.gca()
# plot_distance()
plot_average_time()
