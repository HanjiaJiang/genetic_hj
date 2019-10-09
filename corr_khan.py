import numpy as np
import matplotlib
import microcircuit_tools as tools
# matplotlib.rcParams['font.size'] = 30.0

labels = ['E', 'PV', 'SOM', 'VIP']

def evaluate(result, target):
    t_arr = target.flatten()
    r_arr = result.flatten()
    # take out repeated elements; to be improved
    t_arr = np.concatenate((
        t_arr[0:1], t_arr[4:6], t_arr[8:11], t_arr[12:16]))
    r_arr = np.concatenate((
        r_arr[0:1], r_arr[4:6], r_arr[8:11], r_arr[12:16]))
    sum = 0.0
    cnt = 0
    fitness = 10.0
    for t, r in zip(t_arr, r_arr):
        if np.isnan(t) or np.isnan(r):
            cnt = 0 # as an error flag here
            break
        dif = (t - r) ** 2
        sum += dif
        cnt += 1
    if cnt != 0:
        fitness = np.sqrt(sum/cnt)
    return (fitness,)

# pre-learning
arr_pre = np.array([[0.123, 0.145, 0.112, 0.113],
                [0.145, 0.197, 0.163, 0.193],
                [0.112, 0.163, 0.211, 0.058],
                [0.113, 0.193, 0.058, 0.186]])

# post-learning
arr_post = np.array([[0.075, 0.092, 0.035, 0.092],
                [0.092, 0.144, 0.036, 0.151],
                [0.035, 0.036, 0.204, 0.000],
                [0.092, 0.151, 0.000, 0.176]])


fitness = evaluate(arr_post, arr_pre)[0]
print('fitness (pre vs. post) = {:.3f}'.format(fitness))

tools.interaction_barplot(arr_post, -0.1, 0.25, labels, 'mean corr coef')
