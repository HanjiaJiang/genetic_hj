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
mut_degrees = {'major': 0.3, 'minor': 0.05}    # s.d. of mutation range (unit: times of mean)
set_mut_bound = False
mut_bound = [0.5, 1.5]

np.set_printoptions(precision=4, suppress=True)

# target firing rate
target_arr = np.array([2.7, 13.8, 2.6, 14.6,
                       0.5, 10.2, 2.6,
                       6.8, 7.5, 2.8,
                       6.1, 16.9, 3.9])

# dia_arr_exp = np.array([
#     [100.0, 141.0, 100.0, 0.0, 100.0, 141.0, 0.0, 100.0, 106.0, 106.0, 100.0,
#      0.0, 0.0],
#     [88.00, 141.0, 100.0, 0.0, 141.0, 0.0, 0.0, 106.0, 106.0, 106.0, 0.0, 0.0,
#      0.0],
#     [88.00, 141.0, 0.0, 141.0, 0.0, 0.0, 0.0, 0.0, 106.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#
#     [100.0, 100.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 100.0, 0.0,
#      0.0],
#     [0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#
#     [100.0, 106.0, 106.0, 0.0, 100.0, 0.0, 0.0, 100.0, 141.0, 141.0, 100.0,
#      0.0, 0.0],
#     [106.0, 106.0, 106.0, 0.0, 0.0, 0.0, 0.0, 141.0, 141.0, 141.0, 0.0, 0.0,
#      0.0],
#     [106.0, 106.0, 0.0, 0.0, 0.0, 0.0, 0.0, 141.0, 141.0, 0.0, 0.0, 0.0, 0.0],
#
#     [100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 100.0,
#      100.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# ])

workingdir = os.getcwd()

# origin_probs = np.load('conn_probs_ini.npy')


def create_individual():
    probs = np.array([2000.0, 2000.0, 1500.0, 450.0,
                      2000.0, 2000.0, 1500.0,
                      2000.0, 2000.0, 1500.0,
                      2000.0, 2000.0, 1500.0
                       ])
    # probs = np.concatenate((origin_probs, probs))
    return probs


# fitness
def evaluate(result_fr, target_fr):
    tup = ()
    # distance to target fr
    for t, r in zip(target_fr, result_fr):
        # break if nan appears
        if np.isnan(t) or np.isnan(r):
            for i in range(len(target_fr)):
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
    assert (0 <= p <= 1)
    new = ind
    for i, conn in enumerate(new):
        if np.random.random() <= p and i != 3:
            mut_sd = mut_degrees['minor']
            # if dia_arr_exp[i, j] != 0:
            #     mut_sd = mut_degrees['minor']
            # else:
            #     mut_sd = mut_degrees['major']
            # mutation
            new_conn = conn + mut_sd * conn * (np.random.randn())
            new[i] = new_conn
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
    # divide into groups of 10
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
                            '/output/g={0:02d}_ind={1:02d}/mean_fr.npy'.format(g, map_id)
                ) is False:
                    fin_flag = False
            # break if this generation takes too long
            if time.time() - t0 > (4 * 3600):
                break


# fitness function should minimize the difference between present and target str
creator.create('FitMin', base.Fitness, weights=(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))
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
    ind.fitness.values = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                          100.0, 100.0, 100.0, 100.0, 100.0, 100.0)

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

    # EVALUATION
    for i, ind in enumerate(children):
        fr_file = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'mean_fr.npy'
        if os.path.isfile(fr_file):
            result_arr = np.load(fr_file)
        else:
            print('no result file!')
            result_arr = np.full(13, np.nan)
        ind.fitness.values = box.evaluate(result_arr, target_arr)

    population = combine_pop(population, children)

    print('g{:02d} parents and children fitness values = '.format(g))
    for ind in population:
        print(ind.fitness.values)

    population = box.select(population, n_front)

    population = clone_ind(population, int(N_ind/n_front))

    print('g{:02d} fitness values after selection = '.format(g))
    for ind in population:
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

plt.figure()
plt.plot(np.arange(g), np.sqrt(fitness_evolved[:g, :]/10), 'b.')
# plt.hlines(0.06, 0, g + 10, 'k', linestyles='--', label='pre vs. post RMSE')
# plt.hlines(0, 0, g + 10, 'w', linestyles='--')
plt.legend()
plt.xlabel('Number of generations')
plt.ylabel('Fitness')
plt.title("Evolution of fitness")
plt.tight_layout()
plt.savefig(workingdir + '/output/genetic_hj.png')
