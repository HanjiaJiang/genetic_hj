# Run microcircuit and evaluate the correlation
import os
import sys
import numpy as np
from random import sample
import multiprocessing as mp

# settings
on_server = False
run_sim = True
run_calc = True
stim_times = np.arange(2000.0, 12000.0, 1000.0)
stim_length = 1000.0
bin_width = 125.0

# import microcircuit modules
microcircuit_path = '/home/hanjia/Documents/microcircuit/'
if os.path.isdir(microcircuit_path):
    sys.path.insert(1, microcircuit_path)
import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict
import microcircuit_tools as tools

if on_server:
    cpu_ratio = 1
else:
    cpu_ratio = 0.5
sim_dict['local_num_threads'] = int(mp.cpu_count()*cpu_ratio)


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


def change_parameters(sim_d, net_d, stim_d):
    sim_d['t_sim'] = 12000.0
    net_d['conn_probs'] = np.load('conn_probs.npy')
    net_d['K_ext'] = np.array([3000, 2600, 1200, 500,
                                  2700, 2400, 2800,
                                  1900, 2600, 1300,
                                  2400, 2400, 2100])
    net_d['g'] = 4.0
    net_d['bg_rate'] = 4.0
    stim_d['thalamic_input'] = False
    return sim_d, net_d, stim_d


if run_sim:
    sim_dict, net_dict, stim_dict = change_parameters(sim_dict, net_dict, stim_dict)
    net = network.Network(sim_dict, net_dict, stim_dict)
    net.setup()
    net.simulate()
    tools.plot_raster(
        sim_dict['data_path'], 'spike_detector', 1900.0, 2100.0
    )

if run_calc:
    tmp_arr = network_corr(
        sim_dict['data_path'], 'spike_detector', stim_times, stim_length, bin_width)
    np.save('coef_arr.npy', tmp_arr)
    tools.fire_rate(sim_dict['data_path'], 'spike_detector', 2000.0, 12000.0)
    tools.boxplot(net_dict, sim_dict['data_path'])
else:
    np.save('coef_arr.npy', np.random.random((4, 4)))

coef_arr = np.load('coef_arr.npy')
print(coef_arr)

labels = ['E', 'PV', 'SOM', 'VIP']
tools.interaction_barplot(coef_arr, -0.1, 0.25, labels, 'mean corr coef')
