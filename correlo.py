import numpy as np
import matplotlib.pyplot as plt
import nest
np.set_printoptions(precision=4, suppress=True)

t_sim = 1000.0

# population
N_e = 80
N_i = 20
pop_e = nest.Create('iaf_psc_exp', N_e)
pop_i = nest.Create('iaf_psc_exp', N_i)
poisson = nest.Create('poisson_generator', 10)
nest.SetStatus(poisson, {'rate': 100000.0, 'start': 0.0, 'stop': t_sim })
nest.Connect(poisson, pop_e)
nest.Connect(poisson, pop_i)
syn_e = {'weight': 100.0}
syn_i = {'weight': -400.0}
nest.Connect(pop_e, pop_e, {'rule': 'fixed_total_number', 'N': int(N_e*N_e*0.1)}, syn_spec=syn_e)
nest.Connect(pop_i, pop_i, {'rule': 'fixed_total_number', 'N': int(N_i*N_i*0.2)}, syn_spec=syn_i)
nest.Connect(pop_e, pop_i, {'rule': 'fixed_total_number', 'N': int(N_e*N_i*0.2)}, syn_spec=syn_e)
nest.Connect(pop_i, pop_e, {'rule': 'fixed_total_number', 'N': int(N_i*N_e*0.2)}, syn_spec=syn_i)

# correlomatrix settings
delta_tau = 0.3
correlo_dict = {
    'N_channels': 2,
    'delta_tau': delta_tau,
    'tau_max': delta_tau*2,
    'Tstart': 200.0,
    'Tstop': 800.6
}
nest.SetDefaults('correlomatrix_detector', correlo_dict)
correlo = nest.Create('correlomatrix_detector')

# spike detector
spd = nest.Create('spike_detector')
nest.Connect(pop_e, spd)
nest.Connect(pop_i, spd)

# for i in range(10):
#     nest.Connect([pop_e[i]], correlo, syn_spec={'receptor_type': i})
n1 = pop_e[0]
n2 = pop_e[1]
nest.Connect([n1], correlo, syn_spec={'receptor_type': 0})
nest.Connect([n2], correlo, syn_spec={'receptor_type': 1})


# simulate
nest.Simulate(t_sim)
correlo_arr = nest.GetStatus(correlo)[0]['count_covariance']
n_events = nest.GetStatus(correlo)[0]['n_events']

n_bins = int((correlo_dict['Tstop'] - correlo_dict['Tstart'])/correlo_dict['delta_tau'])
print('n_bins by correlo = {}'.format(n_bins))
corr_result = np.zeros((2, 2))
sum_xy_arr = np.zeros((len(correlo_arr), len(correlo_arr)))
for i in range(len(correlo_arr)):
    for j in range(len(correlo_arr)):
        print('\n{} to {}:'.format(i, j))
        print('count_covariance[0, 1, 2] = {}'.format(correlo_arr[i][j]))
        sum_xy = correlo_arr[i][j][0]   # covariance, x to y
        sum_xx = correlo_arr[i][i][0]   # variance x
        sum_yy = correlo_arr[j][j][0]   # variance y
        sum_x = n_events[i]             # n of spikes x
        sum_y = n_events[j]             # n of spikes y
        print('sum_xy = {}, sum_x = {}, sum_y = {}'.format(sum_xy, sum_x, sum_y))
        corr_coef = (n_bins*sum_xy - sum_x*sum_y)/(np.sqrt((n_bins*sum_xx-sum_x*sum_x)*(n_bins*sum_yy-sum_y*sum_y)))
        sum_xy_arr[i, j] = sum_xy
        corr_result[i, j] = corr_coef
print('\nsum_xy_arr by correlo = \n{}'.format(sum_xy_arr))
print('\nPearson\'s r by correlo =')
print(corr_result)

# data
dSD = nest.GetStatus(spd, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
# plt.plot(ts, evs, ".")
# plt.show()
# plt.close()

# get the spike trains
times = []
evs = evs[(ts>correlo_dict['Tstart']) & (ts<correlo_dict['Tstop'])]
ts = ts[(ts>correlo_dict['Tstart']) & (ts<correlo_dict['Tstop'])]
for i in [n1, n2]:
    t_tmp = []
    for t, s in zip(ts, evs):
        if i == s:
            t_tmp.append(t)
    times.append(t_tmp)

print('\nverifying:')
hists = []
for i, t_list in enumerate(times):
    plt.plot(t_list, np.full(len(t_list), i), ".")
    hist, bin_edges = \
        np.histogram(t_list,
                     n_bins)
                     # (correlo_dict['Tstart'], correlo_dict['Tstop']))  # window of analysis
    hists.append(hist)
    print('n of spikes (train {}) by numpy = {}'.format(i, np.sum(hist)))
print('\nn_bins by numpy = {}'.format(len(hists[0])))

for i, hist1 in enumerate(hists):
    for j, hist2 in enumerate(hists):
        sum_xy_arr[i, j] = np.inner(hist1, hist2)
print('\nsum_xy_arr by numpy = \n{}'.format(sum_xy_arr))

print('\nPearson\'s r by numpy = ')
print(np.corrcoef(hists[0], hists[1]))

plt.ylim((-10, 10))
plt.show()
plt.close()
