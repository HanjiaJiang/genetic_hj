# Run microcircuit and evaluate the correlation
import os
import sys
import numpy as np
from random import sample
import multiprocessing as mp

# import microcircuit modules
cwd = os.getcwd()
microcircuit_path = '/home/hanjia/Documents/microcircuit/'
if os.path.isdir(os.path.join(cwd, '/microcircuit/')):
    microcircuit_path = os.path.join(cwd, '/microcircuit/')
sys.path.insert(1, microcircuit_path)

import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict
from functions import special_dict
import microcircuit_tools as tools

# settings
run_sim = True
run_calc = True
on_server = True

eval_corr = False
eval_fr = True

net_dict['g'] = 4.0
net_dict['bg_rate'] = 4.0
net_dict['animal'] = 'rat'
net_dict['renew_conn'] = False
net_dict['conn_probs'] = np.array([[0.0872, 0.3173, 0.4612, 0.0443, 0.1056, 0.4011, 0.0374, 0.0234, 0.09  , 0.1864, 0.    , 0.    , 0.    ],
       [0.3763, 0.3453, 0.2142, 0.0683, 0.0802, 0.0135, 0.026 , 0.0257, 0.1937, 0.2237, 0.0001, 0.0001, 0.0051],
       [0.2288, 0.4216, 0.0263, 0.2618, 0.0033, 0.0097, 0.0363, 0.0003, 0.0222, 0.018 , 0.    , 0.    , 0.    ],
       [0.0222, 0.0487, 0.0561, 0.027 , 0.0021, 0.0085, 0.0141, 0.0002, 0.0008, 0.0051, 0.    , 0.0001, 0.0047],
       [0.0128, 0.0668, 0.049 , 0.0578, 0.1764, 0.4577, 0.2761, 0.0059, 0.0229, 0.0427, 0.    , 0.0019, 0.0212],
       [0.0329, 0.0132, 0.0188, 0.0438, 0.0937, 0.3487, 0.4068, 0.0078, 0.0228, 0.0389, 0.0011, 0.0024, 0.016 ],
       [0.033 , 0.015 , 0.0198, 0.2618, 0.2906, 0.4432, 0.028 , 0.0087, 0.0263, 0.0384, 0.0016, 0.0019, 0.0198],
       [0.0841, 0.0528, 0.072 , 0.0534, 0.0844, 0.0573, 0.0621, 0.0957, 0.1871, 0.1575, 0.0094, 0.0146, 0.0418],
       [0.0705, 0.1211, 0.0444, 0.0169, 0.0315, 0.025 , 0.0188, 0.0846, 0.3574, 0.2594, 0.0041, 0.0107, 0.0213],
       [0.0998, 0.0072, 0.0089, 0.2618, 0.0343, 0.0248, 0.0209, 0.0587, 0.1182, 0.0373, 0.0054, 0.0122, 0.0262],
       [0.    , 0.0017, 0.0029, 0.007 , 0.0297, 0.0133, 0.0086, 0.0381, 0.0162, 0.0138, 0.021 , 0.3249, 0.3014],
       [0.0026, 0.0001, 0.0002, 0.0019, 0.0047, 0.002 , 0.0004, 0.015 , 0.    , 0.0028, 0.1865, 0.3535, 0.2968],
       [0.0021, 0.    , 0.0002, 0.2618, 0.0043, 0.0018, 0.0003, 0.0141, 0.    , 0.0019, 0.1955, 0.3321, 0.0307]])
net_dict['N_full'] = np.array([5096, 400, 136, 136, 4088, 224, 112, 3264, 360, 320, 4424, 224, 184])

stim_dict['thalamic_input'] = False
stim_dict['th_start'] = np.arange(1500.0, sim_dict['t_sim'], 250.0)
stim_dict['orientation'] = 0.0
# stim_dict['PSP_th'] = 0.15
# stim_dict['PSP_sd'] = 0.1

special_dict['orient_tuning'] = False
special_dict['som_fac'] = True
special_dict['pv_dep'] = True
special_dict['pv2all_dep'] = True
special_dict['weak_dep'] = True

# assign cpu using ratio
if on_server:
    cpu_ratio = 1
else:
    cpu_ratio = 0.5
sim_dict['local_num_threads'] = int(mp.cpu_count() * cpu_ratio)

if eval_corr:
    t_sim = 12000.0
    net_dict['conn_probs'] = np.load('ind.npy')
else:
    t_sim = 3000.0
    net_dict['K_ext'] = np.load('ind.npy')    

stim_times = np.arange(2000.0, t_sim, 1000.0)
stim_length = 1000.0
bin_width = 125.0


# evaluate L2/3 correlations; max 500 cells per population
def network_corr(path, name, stim_ts, stim_len, bin_wid):
    print('start data processing...')
    begin = stim_ts[0]
    end = stim_ts[-1] + stim_len
    data_all, gids = tools.load_spike_times(path, name, begin, end)
    l23_hist_arr = []   # pop x stim x n x bin
    # corr_bin_width = 125.0
    net_coef_arr = np.full((4, 4), np.nan)
    # if population >= 4
    if len(data_all) >= 4:
        # loop population
        for h in range(4):
            if len(data_all[h]) != 0:
                # ids and times of all cells
                pop_nids = data_all[h][:, 0]
                pop_times = data_all[h][:, 1]
                # histogram of all stimuli
                hists_all_stim = [] # stim x n x bin
                # cell list
                ns = list(range(gids[h][0], gids[h][1] + 1))
                # shuffle cell list
                if len(ns) > 500:
                    ns = sample(list(range(gids[h][0], gids[h][1]+1)), 500)
                # collect histograms
                for stim_t in stim_ts:
                    hists = []  # of all cells
                    begin = stim_t
                    end = stim_t + stim_len
                    ids_stim = pop_nids[(pop_times >= begin) & (pop_times < end)]
                    times_stim = \
                        pop_times[(pop_times >= begin) & (pop_times < end)]
                    for n in ns:
                        # spike times of each neuron
                        times = times_stim[ids_stim == n]
                        # if len(times) > 0:
                        # make histogram
                        hist, bin_edges = np.histogram(
                            times,
                            int((end - begin) / bin_wid),  # nr. of bins
                            (begin, end))  # window of analysis
                        hists.append(hist)
                    hists_all_stim.append(hists)

                # subtract mean values to get 'noise' data
                hists_all_stim = hists_all_stim - np.mean(hists_all_stim, axis=0)
                for i, hists in enumerate(hists_all_stim):
                    print('pop {} stim {} sample n = {}'.format(h, i, len(hists)))
                l23_hist_arr.append(hists_all_stim)
            else:
                print('population {} no data'.format(h))
                break

        print('start calculate corr...')
        # calculate corr
        for i, hists_all_stim_1 in enumerate(l23_hist_arr):
            for j, hists_all_stim_2 in enumerate(l23_hist_arr):
                print('pop {} vs. {}'.format(i, j))
                if j >= i:
                    coefs = []
                    if j == i:  # same population
                        for hists in hists_all_stim_1:
                            coef = tools.get_mean_corr(hists)
                            if coef != np.nan:
                                coefs.append(coef)
                    elif j > i: # different population
                        for k, hists in enumerate(hists_all_stim_1):
                            coef = tools.get_mean_corr(hists, hists_all_stim_2[k])
                            if coef != np.nan:
                                coefs.append(coef)
                    net_coef_arr[i, j] = net_coef_arr[j, i] = np.mean(coefs)

    return net_coef_arr


if run_sim:
    net = network.Network(sim_dict, net_dict, stim_dict, special_dict)
    net.setup()
    net.simulate()
    tools.plot_raster(
        sim_dict['data_path'], 'spike_detector', 1900.0, 2100.0
    )

# for testing
coef_arr = np.random.random((4, 4))
mean_fr = np.random.random(13)
# for real
if run_calc:
    mean_fr_cache, std_fr = tools.fire_rate(sim_dict['data_path'], 'spike_detector', 1000.0, t_sim)
    tools.boxplot(net_dict, sim_dict['data_path'])
    if eval_corr:
        coef_arr = network_corr(
            sim_dict['data_path'], 'spike_detector', stim_times, stim_length,
            bin_width)
    mean_fr = mean_fr_cache

np.save('mean_fr.npy', mean_fr)

# output corr results
if eval_corr:
    np.save('coef_arr.npy', coef_arr)
    print('corr. coef. = ')
    print(coef_arr)
    labels = ['E', 'PV', 'SOM', 'VIP']
    tools.interaction_barplot(coef_arr, -0.1, 0.25, labels, 'mean corr coef')

