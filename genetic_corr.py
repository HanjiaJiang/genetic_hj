'''
Genetic algorithm for connectivity map
by Hanjia
'''

import os
import numpy as np
from random import choice, randint
from deap import base, creator, tools
import matplotlib.pyplot as plt
import time
from batch_genetic import batch_genetic

wait_1min = False

N_ind = 20      # number of individuals in a population
p_cx = 0.8      # cross-over probability
p_mut = 0.2     # mutation probability
max_generations = 20
# s.d. of mutation range (unit: times of mean)
mut_degrees = {'major': 0.3, 'minor': 0.05}
set_mut_bound = False
mut_bound = [0.5, 1.5]

np.set_printoptions(precision=4, suppress=True)

target_arr = np.array([[0.123, 0.145, 0.112, 0.113],
                        [0.145, 0.197, 0.163, 0.193],
                        [0.112, 0.163, 0.211, 0.058],
                        [0.113, 0.193, 0.058, 0.186]])

workingdir = os.getcwd()

origin_probs = np.load('conn_probs_ini.npy')


def create_individual():
    return origin_probs


# fitness
def evaluate(ind_map, origin_map, result_corr, target_corr):
    origin_map = origin_map[:4, :4].flatten()
    ind_map = np.array(ind_map)[:4, :4].flatten()
    t_arr = np.array([])
    r_arr = np.array([])
    for i in range(4):
        for j in range(4):
            if j >= i:
                t_arr = np.append(t_arr, target_corr[i, j])
                r_arr = np.append(r_arr, result_corr[i, j])

    # fitness tuple to be returned
    tup = ()

    # deviation from original map
    for conn1, conn2 in zip(ind_map, origin_map):
        dev = np.abs(conn1 - conn2)
        if not tup:
            tup = (dev, )
        else:
            tup = tup + (dev, )

    # distance to target correlations
    for t, r in zip(t_arr, r_arr):
        # break if nan appears
        if np.isnan(t) or np.isnan(r):
            for i in range(26):
                if i == 0:
                    tup = (10,)
                else:
                    tup = tup + (10,)
            break
        dev = np.abs(t - r)
        if not tup:
            tup = (dev, )
        else:
            tup = tup + (dev, )

    return tup


def cxOnePointStr(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = randint(1, size - 1)
    indCls = type(ind1)

    new1, new2 = ind1[:cxpoint], ind2[:cxpoint]
    new1 = np.concatenate((new1, ind2[cxpoint:]))
    new2 = np.concatenate((new2, ind1[cxpoint:]))

    return indCls(new1), indCls(new2)


def mutSNP1(ind, p):
    l23 = np.arange(0, 4)
    assert (0 <= p <= 1)
    new = ind
    for i, row in enumerate(new):
        for j, conn in enumerate(row):
            if np.random.random() <= p:
                if i in l23:    # only when target is in L2/3
                    if j in l23:
                        mut_sd = mut_degrees['major']
                    else:
                        mut_sd = mut_degrees['minor']

                    # mutation
                    new_conn = conn + mut_sd * conn * (np.random.randn())
                    new[i][j] = new_conn
    return type(ind)(new)

def clone_ind(ind_list, times):
    # indCls = type(ind_list[0])
    return_list = []
    if times < 1:
        times = 1
    for t in range(times):
        shuffled = np.arange(len(ind_list))
        np.random.shuffle(shuffled)
        for i in shuffled:
            return_list.append(box.clone(ind_list[i]))
    return return_list

def combine_pop(pop1, pop2):
    indCls = type(pop1[0])
    return_list = []
    for ind1 in pop1:
        return_list.append(box.clone(ind1))
    for ind2 in pop2:
        return_list.append(box.clone(ind2))
    return return_list


def do_and_check(survivors, g):
    # divide into groups of n
    n = 5
    for i in range(int(len(survivors) / n)):
        # do the simulations
        map_ids = np.arange(i * n, (i + 1) * n)
        batch_genetic(survivors[i * n:(i + 1) * n], g, map_ids)
        fin_flag = False  # finish flag
        t0 = time.time()
        # check results
        while fin_flag is False:
            if wait_1min is True:
                time.sleep(60)
            else:
                time.sleep(1)
            fin_flag = True
            for map_id in map_ids:
                if os.path.isfile(
                            workingdir +
                            '/output/g={0:02d}_ind={1:02d}/coef_arr.npy'.format(g, map_id)
                ) is False:
                    fin_flag = False
            # break if this generation takes too long
            if time.time() - t0 > (4 * 3600):
                break



# fitness function should minimize the difference between present and target str
creator.create('FitMin', base.Fitness, weights=(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))
# individual is a list (conn map)
creator.create('Individual', list, fitness=creator.FitMin)  # , n=len(target))

# register functions needed for initialization, evaluation etc.
box = base.Toolbox()
box.register('create_ind', create_individual)
box.register('ind', tools.initIterate, creator.Individual, box.create_ind)
box.register('pop', tools.initRepeat, list, box.ind)

box.register('evaluate', evaluate)
box.register('crossover', cxOnePointStr)
box.register('mutate', mutSNP1)
box.register('select', tools.selNSGA2)
#box.register('select', tools.selTournament, tournsize=3)


### INITIALIZATION
population = box.pop(n=N_ind)
for ind in population:
    ind.fitness.values = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                          10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                          10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                          10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                          10.0, 10.0)

# ### EVOLUTION
g = 0
# fits = [10 for i in population]
fitness_evolved = []
# fitness_evolved = np.zeros((max_generations, 5))
# best5_inds_evolved = np.zeros((max_generations, 5))
n_front = int(N_ind/2)
while g < max_generations:
    ## SELECTION
    children = clone_ind(population, 1)
    # survivors = clone_ind(survivors, int(N_ind/n_front))

    ## GENETIC OPERATIONS
    # crossing-over
    half = int(len(children) / 2)
    chances = np.random.random(size=half)
    chances = np.where(chances <= p_cx)[0]
    for i in chances:
        new1, new2 = box.crossover(children[i], children[i + half])
        children[i] = new1
        children[i + half] = new2
        # new1 and new2 are new instances of Individual class, so there is no
        # need to delete or invalidate their fitness values

    # mutation
    for i, ind in enumerate(children):
        children[i] = box.mutate(ind, p_mut)

    # SIMULATION
    do_and_check(children, g)
    # np.save('survivors.npy', children)

    # EVALUATION
    for i, ind in enumerate(children):
        corr_file = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'coef_arr.npy'
        if os.path.isfile(corr_file):
            result_arr = np.load(corr_file)
        else:
            result_arr = np.full((4, 4), np.nan)
        ind.fitness.values = box.evaluate(ind, origin_probs, result_arr, target_arr)

    print('\ng{:02d} parents = '.format(g))
    for ind in population:
        print(np.sum(ind))
        print(ind.fitness.values)
    population = combine_pop(population, children)
    print('\ng{:02d} parents and children = '.format(g))
    for ind in population:
        print(np.sum(ind))
        print(ind.fitness.values)
    population = box.select(population, n_front)
    population = clone_ind(population, int(N_ind/n_front))
    print('\ng{:02d} population after selection = '.format(g))
    for ind in population:
        print(np.sum(ind))
        print(ind.fitness.values)

    # save fitness values
    tmp = [list(ind.fitness.values) for ind in population]
    fitness_evolved.append(tmp)
    np.save(
        workingdir + '/output/fitness_evolved_g{:02d}.npy'.format(g),
        fitness_evolved)

    # save order by fitness sum
    fitness_sum = [np.sum(i.fitness.values) for i in population]
    print('fitness sums = {}\n'.format(fitness_sum))
    order_by_fitness = np.arange(0, 20)[np.argsort(fitness_sum)]
    np.save(
        workingdir + '/output/order_by_fitness_sum_g{:02d}.npy'.format(g),
        order_by_fitness)
    np.save(
        workingdir + '/output/best_ind_g{:02d}.npy'.format(g),
        population[order_by_fitness[0]])

    # delete .gdf files to save space
    for i in range(N_ind):
        data_dir = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'data/'
        if os.path.isdir(data_dir):
            for item in os.listdir(data_dir):
                if item.endswith('.gdf'):
                    os.remove(data_dir + item)

    g += 1

# plt.figure()
# plt.plot(np.arange(g), np.sqrt(fitness_evolved[:g, :]/10), 'b.')
# # plt.hlines(0.06, 0, g + 10, 'k', linestyles='--', label='pre vs. post RMSE')
# # plt.hlines(0, 0, g + 10, 'w', linestyles='--')
# plt.legend()
# plt.xlabel('Number of generations')
# plt.ylabel('Fitness (RMSE)')
# plt.title("Evolution of 5 best individuals' fitness")
# plt.tight_layout()
# plt.savefig(workingdir + '/output/genetic_hj.png')
