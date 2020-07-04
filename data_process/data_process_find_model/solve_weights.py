import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import torch
from torch import tensor
import csv
import os
import time
import sys

# setting of program
iter_start = 2
iter_end = 10
robot_option = 1

if robot_option == 0:
    robot = 'five_link/'
else:
    if robot_option == 1:
        robot = 'cassie/'

# plot setting
task_space1 = '3d'

dir1 = '../dairlib_data/'+robot+task_space1+'/robot_' + str(robot_option) + \
'_original_iter50/'
label1 = 'Distance between the initial guess and the result'
line_type1 = 'k-'


def solve_weight(iteration, sample):
    # calculate weights for samples from last iteration
    j = 0
    w_length = np.shape(np.genfromtxt(dir1 + str(iteration) + '_' + str(sample) + '_w.csv', delimiter=","))[0]
    w = np.zeros([w_length, 125])
    while os.path.isfile(dir1+str(iteration-1)+'_'+str(j)+'_w.csv'):
        w[:, j] = np.genfromtxt(dir1 + str(iteration-1) + '_' + str(j) + '_w.csv', delimiter=",")
        j = j + 1
    result = np.genfromtxt(dir1 + str(iteration) + '_' + str(sample) + '_w.csv', delimiter=",")
    weights = np.linalg.lstsq(w, result, rcond=None)[0]
    print(weights)
    new_initial_guess = w@weights
    dis = np.linalg.norm(new_initial_guess-result)
    print(dis)

solve_weight(2, 2)
