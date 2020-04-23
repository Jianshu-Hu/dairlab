iter_start = 2
iter_end = 20
robot_option = 0  # 0 is five-link robot. 1 is cassie_fixed_spring

n_sampel_sl = 5  # should be > 0
n_sampel_gi = 5  # should be > 0
N_sample = n_sampel_sl * n_sampel_gi
print('n_sampel_sl = ' + str(n_sampel_sl))
print('n_sampel_gi = ' + str(n_sampel_gi))

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
'_new_2/'
label5 = 'restricted number of sample using initial guess 2 '
line_type5 = 'k:'


