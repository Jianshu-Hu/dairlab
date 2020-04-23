import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import csv
import os
import time
import sys

# setting of program
iter_start = 2
iter_end = 20
robot_option = 0
sample_num = 25

# plot setting
task_space1 = 'uniform_grid'
task_space2 = 'random_sample'

dir1 = '../dairlib_data/'+task_space1+'/robot_' + str(robot_option) + \
'_original/'
label1 = 'uniform grid with original initial guess'
line_type1 = 'k-'

dir2 = '../dairlib_data/'+task_space1+'/robot_' + str(robot_option) + \
'_new_1/'
label2 = 'uniform grid with new initial guess 1'
line_type2 = 'k--'

dir3 = '../dairlib_data/'+task_space1+'/robot_' + str(robot_option) + \
'_new_2/'
label3 = 'uniform grid with new initial guess 2'
line_type3 = 'k:'

dir4 = '../dairlib_data/'+task_space2+'/robot_' + str(robot_option) + \
'_new_2/'
label4 = 'without uniform grid using new initial guess 2'
line_type4 = 'k--'

dir5 = '../dairlib_data/'+task_space2+'/robot_' + str(robot_option) + \
'_restricted_number/'
label5 = 'restricted number of sample using new initial guess 2 '
line_type5 = 'k:'


def distance(is_new_initial_guess, dir, line_type, label_name):
    diff = []

    for i in range(iter_start, iter_end+1):
        for j in range(sample_num):
            if is_new_initial_guess:
                initial_guess = np.genfromtxt(dir+str(i)+'_'+str(j)+'_initial_guess.csv', delimiter=",")
                result = np.genfromtxt(dir+str(i)+'_'+str(j)+'_w.csv', delimiter=",")
            else:
                initial_guess = np.genfromtxt(dir+str(i-1)+'_'+str(j)+'_w.csv', delimiter=",")
                result = np.genfromtxt(dir + str(i) + '_' + str(j) + '_w.csv', delimiter=",")
            diff.append(np.linalg.norm(result-initial_guess))

    ax1.plot(range(len(diff)), diff, line_type, linewidth=3.0, label=label_name)


def average_time(dir, line_type, label_name):
    average_solve_time = []
    for i in range(iter_start, iter_end+1):
        solve_time = []
        j = 0
        while os.path.isfile(dir+str(i)+'_'+str(j)+'_solve_time.csv'):
            solve_time.append(np.genfromtxt(dir+str(i)+'_'+str(j)+'_solve_time.csv', delimiter=","))
            j=j+1
        average_solve_time.append(np.array(solve_time).sum()/len(solve_time))
        # total solve time
        # average_solve_time.append(np.array(solve_time).sum())
    ax1.plot(range(iter_start, iter_end+1), average_solve_time, line_type, linewidth=3.0, label=label_name)

def average_cost(dir, line_type, label_name):
    average_solve_time = []
    for i in range(iter_start, iter_end+1):
        solve_time = []
        j = 0
        while os.path.isfile(dir+str(i)+'_'+str(j)+'_c.csv'):
            if np.genfromtxt(dir+str(i)+'_'+str(j)+'_is_success.csv', delimiter=","):
                solve_time.append(np.genfromtxt(dir+str(i)+'_'+str(j)+'_c.csv', delimiter=","))
                j = j+1
        average_solve_time.append(np.array(solve_time).sum()/len(solve_time))
    ax1.plot(range(iter_start, iter_end+1), average_solve_time, line_type, linewidth=3.0, label=label_name)


def solve_weight(num):
    iter = 3
    sample = num
    result_current = np.genfromtxt(dir2+str(iter)+'_'+str(sample)+'_w.csv', delimiter=",")
    length = result_current.shape[0]
    result_past = np.zeros([length, sample_num])

    norm_weight = np.zeros(sample_num)
    sl_old = np.zeros(sample_num)
    gi_old = np.zeros(sample_num)
    sl = float(np.genfromtxt(dir2 + str(iter) + '_' + str(sample) + '_stride_length.csv', delimiter=","))
    print('stride length of current sample', sl)
    gi = float(np.genfromtxt(dir2 + str(iter) + '_' + str(sample) + '_ground_incline.csv', delimiter=","))
    print('ground incline of current sample', gi)

    for j in range(sample_num):
        result_past[:, j] = np.genfromtxt(dir2 + str(iter-1) + '_' + str(j) + '_w.csv', delimiter=",")
        sl_old[j] = np.genfromtxt(dir2+str(iter-1)+'_'+str(j)+'_stride_length.csv', delimiter=",")
        gi_old[j] = np.genfromtxt(dir2+str(iter-1)+'_'+str(j)+'_ground_incline.csv', delimiter=",")

    # solve over-constrained equation
    A = np.matmul(np.transpose(result_past), result_past)
    b = np.matmul(np.transpose(result_past), result_current)
    weight = solve(A, b)
    print(np.linalg.norm(np.matmul(result_past,weight)-result_current))
    print(weight)
    # print(gi_old)
    # print(sl_old)
    norm_weight = 1/(np.square((sl_old-sl) / 2 / delta_dist) + np.square((gi_old-gi) / 2 / delta_incline))
    norm_weight = norm_weight/norm_weight.sum()
    print(norm_weight)

def plot_distance():
    # plot distance
    distance(False, dir1, line_type1, label1)
    distance(True, dir2, line_type2, label2)
    distance(True, dir3, line_type3, label3)
    plt.xlabel('Sample Number')
    plt.ylabel('Distance between initial guess and result')
    plt.legend()
    plt.show()

def plot_average_time():
    # plot average solve time
    average_time(dir1, line_type1, label1)
    average_time(dir2, line_type2, label2)
    average_time(dir3, line_type3, label3)
    plt.xlabel('Iteration')
    plt.ylabel('Average solve time for trajectory optimization in one iteration')
    plt.legend()
    plt.show()

def plot_average_cost():
    # plot average cost
    average_cost(dir1, line_type1, label1)
    average_cost(dir2, line_type2, label2)
    average_cost(dir3, line_type3, label3)
    plt.xlabel('Iteration')
    plt.ylabel('Average cost of samples')
    plt.legend()
    plt.show()

fig1 = plt.figure(num=1, figsize=(6.4, 4.8))
ax1 = fig1.gca()
# plot_distance()
plot_average_time()
# plot_average_cost()
