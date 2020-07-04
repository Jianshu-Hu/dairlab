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
    sl = tensor(np.genfromtxt(dir1 + str(iteration) + '_' + str(sample) + '_stride_length.csv', delimiter=","))
    gi = tensor(np.genfromtxt(dir1 + str(iteration) + '_' + str(sample) + '_ground_incline.csv', delimiter=","))
    tr = tensor(np.genfromtxt(dir1 + str(iteration) + '_' + str(sample) + '_turning_rate.csv', delimiter=","))
    gamma = tensor([sl, gi, tr])
    # set weights for samples from last iteration
    j = 0
    weights = torch.zeros(125)
    w_length = np.shape(np.genfromtxt(dir1 + str(iteration) + '_' + str(sample) + '_w.csv', delimiter=","))[0]
    w = torch.zeros([w_length, 125])
    while os.path.isfile(dir1+str(iteration-1)+'_'+str(j)+'_w.csv'):
        sl_old = tensor(np.genfromtxt(dir1 + str(iteration-1) + '_' + str(j) + '_stride_length.csv', delimiter=","))
        gi_old = tensor(np.genfromtxt(dir1 + str(iteration-1) + '_' + str(j) + '_ground_incline.csv', delimiter=","))
        tr_old = tensor(np.genfromtxt(dir1 + str(iteration-1) + '_' + str(j) + '_turning_rate.csv', delimiter=","))
        gamma_old = tensor([sl_old, gi_old, tr_old])

        std = tensor(np.array([10/(sl_max-sl_min), 10/(gi_max-gi_min), 10/(tr_max-tr_min)]))
        # create feature sets
        b = torch.abs(gamma-gamma_old)*std
        b2 = torch.pow(b, 2)
        b3 = torch.pow(b, 3)
        b4 = torch.pow(b, 4)
        features = torch.cat((b, b2, b3, b4), 0)

        total_dif = (features*scale).sum()
        weights[j] = 1/total_dif
        w[:, j] = tensor(np.genfromtxt(dir1 + str(iteration-1) + '_' + str(j) + '_w.csv', delimiter=","))
        j = j + 1
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
    theta_length = np.shape(np.genfromtxt(dir1 + '2_theta_s.csv', delimiter=","))[0]
    C = tensor(np.ones(12), requires_grad=True)
    # stochastic gradient
    lr = 0.01
    momentum = 0.8
    epoch = 1
    last_grad = torch.zeros([12])
    all_loss = []
    for epoch_num in range(1, epoch+1):
        for i in range(iter_start, iter_end+1):
            aver_loss = calculate_aver_loss(3, C)
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
