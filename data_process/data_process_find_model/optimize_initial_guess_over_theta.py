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
iter_end = 50
robot_option = 1
sl_min = 0.1625
sl_max = 0.2375
gi_min = -0.125
gi_max = 0.125
tr_min = -0.3125
tr_max = 0.3125

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


def calculate_initial_guess(iteration, sample, scale):
    # data for current sample
    theta_s = tensor(np.genfromtxt(dir1 + str(iteration) + '_theta_s.csv', delimiter=","))
    theta_sDDot = tensor(np.genfromtxt(dir1 + str(iteration) + '_theta_sDDot.csv', delimiter=","))
    theta = torch.cat((theta_s, theta_sDDot), 0)
    # set weights for samples from last iteration
    j = 1
    num_iters_to_use = 2
    weights = torch.zeros(num_iters_to_use)
    w_length = np.shape(np.genfromtxt(dir1 + str(iteration) + '_' + str(sample) + '_w.csv', delimiter=","))[0]
    w = torch.zeros([w_length, num_iters_to_use])
    for j in range(num_iters_to_use):
        theta_s_old = tensor(np.genfromtxt(dir1 + str(iteration-j-1) + '_theta_s.csv', delimiter=","))
        theta_sDDot_old = tensor(np.genfromtxt(dir1 + str(iteration - j - 1) + '_theta_sDDot.csv', delimiter=","))
        theta_old = torch.cat((theta_s_old, theta_sDDot_old), 0)
        # create feature sets
        b = torch.abs(theta-theta_old)
        b2 = torch.pow(b, 2)
        features = torch.cat((b, b2), 0)

        total_dif = (b2*scale).sum()
        weights[j] = 1/total_dif
        w[:, j] = tensor(np.genfromtxt(dir1 + str(iteration-j-1) + '_' + str(sample) + '_w.csv', delimiter=","))
    weights = weights/weights.sum()
    initial_guess = torch.matmul(w, weights)
    return initial_guess


def calculate_aver_loss(iteration, scale):
    loss = torch.zeros([125])
    j = 0
    while os.path.isfile(dir1+str(iteration)+'_'+str(j)+'_w.csv'):
        initial_guess = calculate_initial_guess(iteration, j, scale)
        result = tensor(np.genfromtxt(dir1 + str(iteration) + '_' + str(j) + '_w.csv', delimiter=","))
        loss[j] = torch.norm(initial_guess-result)
        j = j+1
    aver_loss = loss.sum()/125
    # np.savetxt(str(iteration)+'_loss.csv', loss.detach().numpy(), delimiter=',')
    return aver_loss


def main():
    # initialize C
    theta_length = np.shape(np.genfromtxt(dir1 + '2_theta_s.csv', delimiter=","))[0] + \
                np.shape(np.genfromtxt(dir1 + '2_theta_sDDot.csv', delimiter=","))[0]
    C = tensor(np.ones(theta_length), requires_grad=True)
    # stochastic gradient
    lr = 0.05
    momentum = 0.8
    epoch = 1
    last_grad = torch.zeros([theta_length])
    all_loss = []
    for epoch_num in range(1, epoch+1):
        for i in range(iter_start, iter_end+1):
            aver_loss = calculate_aver_loss(5, C)
            print(i, 'aver loss', aver_loss)
            all_loss.append(aver_loss)
            aver_loss.backward()
            with torch.no_grad():
                current_grad = C.grad.clone().detach()
                C = C-lr*(momentum*last_grad+current_grad)
                last_grad = current_grad.clone().detach()
                np.savetxt('scale/'+str(epoch_num)+'_'+str(i)+'_C.csv', C.detach().numpy(), delimiter=',')
            C.requires_grad_(True)

    all_loss = np.array(all_loss)
    fig1 = plt.figure(num=1, figsize=(6.4, 4.8))
    ax1 = fig1.gca()
    ax1.plot(range(1, len(all_loss) + 1), all_loss, line_type1, linewidth=3.0, label=label1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
