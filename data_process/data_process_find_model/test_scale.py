import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import torch
from torch import tensor
import csv
import os

from optimize_initial_guess_over_theta import calculate_aver_loss
import optimize_initial_guess_over_theta

# C = tensor(np.diag(np.array([0.9, 1.0, 2.0])), requires_grad=True)
C = tensor(np.genfromtxt('scale/1_14_C.csv', delimiter=","), requires_grad=True)
print(C)

aver_loss = calculate_aver_loss(10, C)
print(aver_loss)
