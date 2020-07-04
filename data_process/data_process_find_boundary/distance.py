import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve
import torch
from torch import tensor
import csv
import os


robot_option = 0
if robot_option == 0:
    robot = 'five_link/'
else:
    if robot_option == 1:
        robot = 'cassie/'

dir1 = '../dairlib_data/find_boundary/'+robot+'robot_' + str(robot_option) + \
'_2d/'
label1 = 'use result from last iteration'
line_type1 = 'k-'

label2 = 'interpolated initial guess'
line_type2 = 'k--'


def distance(is_new_initial_guess, dir, line_type, label_name):
    diff = []
    i = 1
    while os.path.isfile(dir+str(i)+'_'+str(0)+'_w.csv'):
        if is_new_initial_guess:
            initial_guess = np.genfromtxt(dir+str(i)+'_'+str(0)+'_initial_guess.csv', delimiter=",")
        else:
            initial_guess = np.genfromtxt(dir+str(i-1)+'_'+str(0)+'_w.csv', delimiter=",")
        result = np.genfromtxt(dir + str(i) + '_' + str(0) + '_w.csv', delimiter=",")
        diff.append(np.linalg.norm(result-initial_guess))
        i = i+1

    ax1.plot(range(1, len(diff)+1), diff, line_type, linewidth=3.0, label=label_name)


def plot_distance():
    # plot distancn
    # n = 125
    # distance(False, dir1, line_type1, label1)
    distance(True, dir1, line_type2, label2)
    plt.xlabel('Sample')
    plt.ylabel('Distance between initial guess and result')
    plt.legend()
    plt.show()


fig1 = plt.figure(num=1, figsize=(6.4, 4.8))
ax1 = fig1.gca()
plot_distance()
