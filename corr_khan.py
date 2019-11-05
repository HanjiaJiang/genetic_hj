import numpy as np
import matplotlib
#import microcircuit_tools as tools
# matplotlib.rcParams['font.size'] = 30.0

labels = ['E', 'PV', 'SOM', 'VIP']

def evaluate(r_arr, t_arr):
    tup = ()
    break_flag = False
    assert type(r_arr) == np.ndarray
    assert type(t_arr) == np.ndarray
    assert r_arr.shape == t_arr.shape
    for i, row in enumerate(r_arr):
        for j, r in enumerate(row):
            t = t_arr[i, j]
            if j >= i:  # take only diagonal + upper triangle
                # break if nan appears
                if np.isnan(r) or np.isnan(t):
                    tup = np.nan
                    break_flag = True
                    break
                dev = r - t
                if not tup:
                    tup = (dev, )
                else:
                    tup = tup + (dev, )                   
        if break_flag is True:
            break
    return tup

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


fitness = evaluate(arr_post, arr_pre)
fitness = np.sqrt(np.mean(np.square(fitness)))
#fitness = np.mean(np.abs(fitness))
print('fitness (pre vs. post) = {:.3f}'.format(fitness))

#tools.interaction_barplot(arr_post, -0.1, 0.25, labels, 'mean corr coef')
