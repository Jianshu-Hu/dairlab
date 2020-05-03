import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import csv
import os
import time
import sys

# setting of program
iter_start = 1
iter_end = 500
robot_option = 0
sample_num = 25

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

dir1 = '../dairlib_data/'+robot+task_space1+'/robot_' + str(robot_option) + \
'_original/'
label1 = 'uniform grid with original initial guess'
line_type1 = 'k-'

dir2 = '../dairlib_data/'+robot+task_space1+'/robot_' + str(robot_option) + \
'_new_1/'
label2 = 'uniform grid with new initial guess 1'
line_type2 = 'k--'

dir3 = '../dairlib_data/'+robot+task_space1+'/robot_' + str(robot_option) + \
'_new_2/'
label3 = 'uniform grid with new initial guess 2'
line_type3 = 'k:'

dir4 = '../dairlib_data/'+robot+task_space2+'/robot_' + str(robot_option) + \
'_new_2/'
label4 = 'using new initial guess 2 without uniform grid '
line_type4 = 'k--'

dir5 = '../dairlib_data/'+robot+task_space2+'/robot_' + str(robot_option) + \
'_restricted_number/'
label5 = 'restricted number of sample using new initial guess 2 '
line_type5 = 'k:'

dir6 = '../dairlib_data/'+robot+task_space2+'/robot_' + str(robot_option) + \
'_iter500/'
label6 = 'restricted number of sample using new initial guess 2 '
line_type6 = 'k:'

dir7 = '../dairlib_data/'+robot+task_space1+'/robot_' + str(robot_option) + \
'_100iter/'
label7 = 'uniform grid with original initial guess'
line_type7 = 'k-'

print("check the dir name and plot setting carefully")

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
            j = j+1
        average_solve_time.append(np.array(solve_time).sum()/len(solve_time))
        # total solve time
        # average_solve_time.append(np.array(solve_time).sum())
    ax1.plot(range(iter_start, iter_end+1), average_solve_time, line_type, linewidth=3.0, label=label_name)


def average_cost(dir, line_type, label_name, normalized, dir_nominal):
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
        cost = []
        j = 0
        while os.path.isfile(dir+str(i)+'_'+str(j)+'_c.csv'):
            if np.genfromtxt(dir+str(i)+'_'+str(j)+'_is_success.csv', delimiter=","):
                cost.append(np.genfromtxt(dir+str(i)+'_'+str(j)+'_c.csv', delimiter=","))
            j = j+1
        aver_cost.append(np.array(cost).sum()/len(cost)/nominal_cost)

    ax1.plot(range(iter_start, iter_end + 1), aver_cost, line_type, linewidth=3.0, label=label_name)


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
    average_time(dir4, line_type4, label4)
    average_time(dir5, line_type5, label5)
    # average_time(dir6, line_type6, label6)
    plt.xlabel('Iteration')
    plt.ylabel('Average solve time for trajectory optimizations in one iteration')
    plt.legend()
    plt.show()

def plot_average_cost():
    # plot average cost
    normalized_with_norminal_cost = 1
    average_cost(dir1, line_type1, label1, normalized_with_norminal_cost, dir1)
    average_cost(dir4, line_type4, label4, normalized_with_norminal_cost, dir4)
    average_cost(dir5, line_type5, label5, normalized_with_norminal_cost, dir1)
    # average_cost(dir6, line_type6, label6, normalized_with_norminal_cost)
    plt.xlabel('Iteration')
    if normalized_with_norminal_cost == 1:
        plt.ylabel('Normalized average cost of samples')
    else:
        plt.ylabel('Average cost of samples')
    plt.legend()
    plt.show()

fig1 = plt.figure(num=1, figsize=(6.4, 4.8))
ax1 = fig1.gca()
# plot_distance()
# plot_average_time()
plot_average_cost()