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
max_generations = 3
# s.d. of mutation range (unit: times of mean)
mut_degrees = {'major': 0.05, 'minor': 0.01}
set_mut_bound = False
mut_bound = [0.5, 1.5]
n_round = 5
nonsense = 1.0e3

target_arr = np.array([2.7, 13.8, 2.6, 14.6,
                       0.5, 10.2, 2.6,
                       6.8, 7.5, 2.8,
                       6.1, 16.9, 3.9])

workingdir = os.getcwd()

outputdir = os.path.join(workingdir, 'output')
if os.path.isdir(outputdir) is False:
    os.mkdir(outputdir)
os.system('cp *.py *.sh ' + outputdir) 

origin_probs = np.load('K_ext.npy')


def create_individual():
    return origin_probs


# values for fitness: deviation of result from target
def evaluate(r_arr, t_arr):
    tup = ()
    assert type(r_arr) == np.ndarray
    assert type(t_arr) == np.ndarray
    assert r_arr.shape == t_arr.shape
    for r, t in zip(r_arr, t_arr):
        if np.isnan(r) or np.isnan(t):
            tup = np.nan
            break
        dev = r - t
        if not tup:
            tup = (dev,)
        else:
            tup = tup + (dev,)
    return tup


def cross_conn(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = randint(1, size - 1)
    indCls = type(ind1)

    new1, new2 = ind1[:cxpoint], ind2[:cxpoint]
    new1 = np.concatenate((new1, ind2[cxpoint:]))
    new2 = np.concatenate((new2, ind1[cxpoint:]))

    return indCls(new1), indCls(new2)


def mut_conn(ind, p):
    assert (0 <= p <= 1)
    new = ind
    for i, conn in enumerate(ind):
        if np.random.random() <= p:
            if i == 3:
                mut_sd = mut_degrees['minor']
            else:
                mut_sd = mut_degrees['major']
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
    # divide into groups of n
    n = n_round
    ok_flg = True
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
                time.sleep(5)
            fin_flag = True
            for map_id in map_ids:
                if os.path.isfile(
                            workingdir +
                            '/output/g={0:02d}_ind={1:02d}/mean_fr.npy'.format(g, map_id)
                ) is False:
                    fin_flag = False
            # break if this generation takes too long
            if time.time() - t0 > (4 * 3600):
                ok_flg = False
                break
    return ok_flg


# fitness function should minimize the difference between present and target
creator.create('FitMin', base.Fitness, weights=(-1, ))
# creator.create('FitMin', base.Fitness, weights=(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1))
# individual is a list (conn map)
creator.create('Individual', list, fitness=creator.FitMin)  # , n=len(target))

# register functions needed for initialization, evaluation etc.
box = base.Toolbox()
box.register('create_ind', create_individual)
box.register('ind', tools.initIterate, creator.Individual, box.create_ind)
box.register('pop', tools.initRepeat, list, box.ind)

box.register('evaluate', evaluate)
box.register('crossover', cross_conn)
box.register('mutate', mut_conn)
# box.register('select', tools.selNSGA2)
box.register('select', tools.selTournament, tournsize=3)


### INITIALIZATION
population = box.pop(n=N_ind)
for ind in population:
    ind.fitness.values = (nonsense, )

# ### EVOLUTION
g = 0
fitness_evolved = []
n_selected = int(N_ind/4)
while g < max_generations:
    np.set_printoptions(precision=4, suppress=True)

    ## clone to children
    children = clone_ind(population, 1)

    print('\ng{:02d} children before reproduction = '.format(g))
    for i, ind in enumerate(children):
        print('{}. {}'.format(i, np.sum(ind)))
        print(ind.fitness.values)

    ## GENETIC OPERATIONS
    # crossing-over
    half = int(len(children) / 2)
    chances = np.random.random(size=half)
    chances = np.where(chances <= p_cx)[0]
    for i in chances:
        new1, new2 = box.crossover(children[i], children[i + half])
        children[i] = new1
        children[i + half] = new2

    # mutation
    for i, ind in enumerate(children):
        children[i] = box.mutate(ind, p_mut)

    print('\ng{:02d} children after reproduction = '.format(g))
    for i, ind in enumerate(children):
        print('{}. {}'.format(i, np.sum(ind)))
        print(ind.fitness.values)

    # SIMULATION
    flg = do_and_check(children, g)
    if flg is False:
        print('Failure in g={0:02d}'.format(g))

    # EVALUATION
    # errors_list = []
    for i, ind in enumerate(children):
        result_file = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'mean_fr.npy'
        if os.path.isfile(result_file):
            result_arr = np.load(result_file)
        else:
            result_arr = np.full(target_arr.shape, np.nan)
        errors = box.evaluate(result_arr, target_arr)
        # errors_list.append(np.array(errors))
        if errors == np.nan:
            ind.fitness.values = (nonsense, )
            print('fit_values == np.nan')
        else:
            ind.fitness.values = (np.sqrt(np.mean(np.square(errors))), )
            print('fit_values = \n{}'.format(errors))

    # save and print
    fitness_evolved.append([ind.fitness.values[0] for ind in children])
    values = [ind.fitness.values[0] for ind in children]
    order_by_fitness = np.arange(0, N_ind)[np.argsort(values)]
    # np.save(workingdir +'/output/errors_g{:02d}.npy'.format(g), np.array(errors_list))
    np.save(
        workingdir + '/output/population_selected_g{:02d}.npy'.format(g),
        population)
    np.save(
        workingdir + '/output/fitness_evolved_g{:02d}.npy'.format(g),
        fitness_evolved)
    np.save(
        workingdir + '/output/order_by_fitness_g{:02d}.npy'.format(g),
        order_by_fitness)

    # print('\ng{:02d} population before combining = '.format(g))
    # for i, ind in enumerate(population):
    #     print('{}. {}'.format(i, np.sum(ind)))
    #     print(ind.fitness.values)

    # combine and select
    # population = combine_pop(population, children)

    # print('\ng{:02d} population after combining = '.format(g))
    # for i, ind in enumerate(population):
    #     print('{}. {}'.format(i, np.sum(ind)))
    #     print(ind.fitness.values)

    population = box.select(children, n_selected)

    print('\ng{:02d} population after selection = '.format(g))
    for i, ind in enumerate(population):
        print('{}. {}'.format(i, np.sum(ind)))
        print(ind.fitness.values)

    # clone for next generation
    population = clone_ind(population, int(N_ind/n_selected))
    print('\ng{:02d} population after cloning = '.format(g))
    for i, ind in enumerate(population):
        print('{}. {}'.format(i, np.sum(ind)))
        print(ind.fitness.values)

    # delete .gdf files to save space
    for i in range(N_ind):
        data_dir = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'data/'
        if os.path.isdir(data_dir):
            for item in os.listdir(data_dir):
                if item.endswith('.gdf'):
                    os.remove(data_dir + item)

    g += 1

plt.figure()
plt.plot(np.arange(len(fitness_evolved[3:])), fitness_evolved[3:], 'b.')
plt.hlines(0.06, 0, g + 10, 'k', linestyles='--', label='RMSE of (pre vs. post)')
plt.hlines(0, 0, g + 10, 'w', linestyles='--')
plt.legend()
plt.xlabel('Number of generations')
plt.ylabel('Fitness')
plt.title("Evolution of fitness")
plt.tight_layout()
plt.savefig(workingdir + '/output/genetic_hj.png')
