import numpy as np
import ee.eeutility as util
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random


#def dca_simple_evaluation(p, length, cf):
#    q = 1-p
#    correct = (1-q**length) / float(1-q)
#    return correct/length


def dca_evaluation(transitions, lengths, evaluator, energy_costs=1, energy_off=0, confusion=None, active=1,
                   prob=None, sleeping_exp=None, working_exp=None, max_length=None):
    if type(lengths) == int:
        lengths = [lengths]*len(transitions)
    if type(energy_costs) == int:
        energy_costs = [energy_costs]*len(transitions)
    if confusion is None:
        confusion = [[1 if i == j else 0 for j in range(len(lengths))] for i in range(len(lengths))]
    #confusion = util.normalizeMatrix(confusion, rows=True)
    if prob is None:
        if max_length is None:
            max_length = max(lengths)+1
        prob, sleeping_exp, working_exp = precalculate(transitions, max_length, active)
    matrix, spared, avg_energy = _duty_predict(transitions, lengths, prob, sleeping_exp, working_exp,
                                               confusion=confusion, active=active,
                                               energy_costs=energy_costs)
    quality = evaluator(matrix)
    energy = _duty_energy(spared+1, energy_off, avg_energy)
    return quality, energy


def dca_find_tradeoffs(length, maxCycle, evaluator, NGEN=200, gen_size=200, indpb=0.05, seeded=False,
                       save_history=False, verbose=False):
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 1, maxCycle)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _mutate, indpb=indpb, length=maxCycle)
    toolbox.register("select", tools.selNSGA2)
    MU = gen_size
    LAMBDA = gen_size * 2
    CXPB = 0.7
    MUTPB = 0.2
    pop = toolbox.population(n=MU)
    if seeded:
        for i in range(length):
            pop[0][i] = 1
            pop[1][i] = maxCycle
    hof = tools.ParetoFront()
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, halloffame=hof, verbose=verbose)
    return hof


def precalculate(transition, max_length, active=1):
    l = len(transition)
    expected_sleeping_all = {}
    expected_active_all = {}
    probs_all = []
    for true in range(l):
        start = np.eye(true+1,l)[true]
        probs =  np.array([start] + list(_reduce_opposite(lambda x: np.dot(x,transition), start, active+max_length-1)))
        probs_all.append(probs)
        expected_sleeping = np.zeros(l).T
        for i in range(max_length):
            expected_sleeping += probs[i]
            expected_sleeping_all[(true,i)] = np.array(expected_sleeping)
            expected_active = np.sum(probs[i+1:i+active], axis=0)
            expected_active_all[(true,i)] = np.array(expected_active)
    return probs_all, expected_sleeping_all, expected_active_all


def _mutate(sequence, indpb=0.05, length = 30):
    for i in range(len(sequence)):
        if random.random()<indpb:
            sequence[i] = random.randint(1, length)
    return (sequence,)


def _reduce_opposite(fn, start, n_times=3):
    for _ in range(n_times):
        start = fn(start)
        yield start


def _duty_predict(transition, lengths, prob, sleeping_exp, working_exp, confusion=None, active=1,
                  energy_costs=None):
    if confusion is None:
        confusion = [[1 if i == j else 0 for j in range(len(lengths))] for i in range(len(lengths))]
    confusion = np.array(confusion)

    length = len(transition)
    next_state_prob = []

    for t_from in range(length):
        for p_from in range(length):
            probs = prob[t_from][lengths[p_from] + active - 1]
            next_states = np.multiply(confusion.T, probs).T
            next_state_prob.append(next_states.flatten())

    distribution = util.get_steady_state_trans(np.array(next_state_prob)).reshape((length, length))

    matrix = np.zeros((length, length))
    for true in range(length):
        for pred in range(length):
            matrix[true][pred] = sum(
                [distribution[i][pred] * sleeping_exp[(i, lengths[pred] - 1)][true] for i in range(length)])
    if active > 1:
        for cycle_true in range(length):
            for cycle_pred in range(length):
                matrix += distribution[cycle_true][cycle_pred] * np.multiply(confusion.T, working_exp[
                    (cycle_true, lengths[cycle_pred] - 1)]).T

    matrix /= np.sum(matrix)

    avg_energy = 0
    if energy_costs is not None:
        if type(energy_costs[0]) == list:
            avg_energy = np.sum(np.multiply(distribution.reshape((length, length)), np.array(energy_costs).T))
        else:
            avg_energy = np.dot(np.sum(distribution.reshape((length, length)), axis=0), energy_costs)
    spared = np.dot(np.sum(distribution.reshape((length, length)), axis=0), lengths)
    spared = (spared - 1) / active

    return matrix, spared, avg_energy


def _duty_energy(gain, base_off, base_on):
    prop_on = 1 / float(gain)
    prop_off = (gain-1) / float(gain)
    return prop_off*base_off + prop_on*base_on
