import sys
import numpy as np
import matplotlib.pyplot as plt
from random import sample
on_server = True
if not on_server:
    sys.path.insert(1, '/home/hanjia/Documents/microcircuit/')
import network
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict
from scan_params import *
import microcircuit_tools as tools
import multiprocessing as mp

run_sim = True
run_calc = True
stim_times = np.arange(2000.0, 12000.0, 1000.0)
stim_length = 1000.0
bin_width = 125.0

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


net_dict['conn_probs'] = np.load('conn_probs.npy')
if on_server:
    cpu_ratio = 1
else:
    cpu_ratio = 0.5
sim_dict['local_num_threads'] = int(mp.cpu_count()*cpu_ratio)
sim_dict['t_sim'] = 12000.0
net_dict['K_ext'] = np.array([2000, PV_ext_scan, SOM_ext_scan, VIP_ext_scan,
                              2000, PV_ext_scan, SOM_ext_scan,
                              2000, PV_ext_scan, SOM_ext_scan,
                              2000, PV_ext_scan, SOM_ext_scan])
net_dict['g'] = g_scan
net_dict['bg_rate'] = bg_scan
stim_dict['thalamic_input'] = False

if run_sim:
    net = network.Network(sim_dict, net_dict, stim_dict)
    net.setup()
    net.simulate()
    tools.plot_raster(
        sim_dict['data_path'], 'spike_detector', 1900.0, 2100.0
    )
    plt.close()

if run_calc:
    tmp_arr = network_corr(
        sim_dict['data_path'], 'spike_detector', stim_times, stim_length, bin_width)
    np.save('coef_arr.npy', tmp_arr)
    tools.fire_rate(sim_dict['data_path'], 'spike_detector', 2000.0, 12000.0)
    tools.boxplot(net_dict, sim_dict['data_path'])

coef_arr = np.load('coef_arr.npy')
print(coef_arr)

labels = ['E', 'PV', 'SOM', 'VIP']
tools.interaction_barplot(coef_arr, -0.1, 0.25, labels, 'mean corr coef')
# x = np.arange(len(labels))
# barwidth = 0.1
# fig, ax = plt.subplots(figsize=(12, 12))
# for i in range(4):
#     ax.bar(x + barwidth*(i - 1.5), coef_arr[i, :], barwidth, label=labels[i])
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.ylim((-0.1, 0.25))
# fig.tight_layout()
# plt.savefig('corr_model.png')
# plt.show()
