import numpy as np
import ee.eeutility as util
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random


def sca_evaluation(transitions, confusions, energies, quality_fn, matrix_energy=False):
    markov_chain = _transition_matrix(transitions, confusions)
    steady_state = util.get_steady_state_trans(markov_chain)
    steady_state_square = np.reshape(steady_state, (int(len(steady_state)**0.5), int(len(steady_state)**0.5)))
    if not (steady_state_square > -0.00001).all():
        return -float("inf"), float("inf")
    quality = quality_fn(steady_state_square)
    energy = _energy(steady_state_square, energies, matrix_energy)
    return quality, energy


def sca_simple_evaluation(proportions, confusions, energies, quality_fn):
    conf = [proportions[i] * np.array(confusions[i]) for i in range(len(proportions))]
    quality = quality_fn(conf)
    energy = np.dot(energies, proportions.T)
    return quality, energy


def sca_find_tradeoffs(setting_type, num_settings, num_contexts, optimizer, NGEN=50, MU=50, cstree=False):
    if hasattr(creator, 'Fitness'):
        del creator.Fitness
    if hasattr(creator, 'Individual'):
        del creator.Individual
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    if setting_type == "binary":
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                              toolbox.attr_bool, num_settings * num_contexts)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("evaluate", lambda x: optimizer.sca_model(_list_to_config(x, num_contexts))[0])
    if setting_type == "enumerate":
        toolbox.register("attr_int", random.randint, 0, num_settings-1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                              toolbox.attr_int, num_contexts)
        toolbox.register("mutate", _mutate_list, indpb=0.05, max_length=num_settings-1)
        toolbox.register("evaluate", lambda x: optimizer.sca_model(_list_to_config(x, num_contexts),
                                                                   encrypted=True, cstree=cstree)[0])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selNSGA2)
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, halloffame=hof, verbose=False)
    return [_list_to_config(h, num_contexts) for h in hof]


def _mutate_list(sequence, indpb=0.05, max_length=0):
    if max_length == 0:
        print("Error: the index length is 0")
    for i in range(len(sequence)):
        if random.random() < indpb:
            sequence[i] = random.randint(0, max_length)
    return (sequence,)


def _list_to_config(lst, num_contexts):
    config = np.reshape(lst, (num_contexts, -1))
    config = [(tuple(s) if len(s) > 1 else s[0]) for s in config]
    return config


def _transition_matrix(transitions, confusions):
    length = len(confusions)
    len_square = len(confusions)**2
    chain = [[0]*len_square for i in range(len_square)]
    for i in range(len_square):
        for j in range(len_square):
            from_r, to_r, from_p, to_p = i//length, j//length, i%length, j%length
            prob_a = transitions[from_r][to_r]
            prob_b = confusions[from_p][to_r][to_p]
            chain[i][j] = prob_a*prob_b
    return np.array(chain)


def _energy(steady_state_square, energies, matrix_energy):
    steady_sum = np.sum(list(map(lambda x: np.real(x), steady_state_square)), axis=0).T
    if not matrix_energy:
        return np.dot(energies, steady_sum)
    else:
        return np.sum(np.multiply(steady_state_square, np.array(energies).T))
