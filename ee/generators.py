import numpy as np
import pandas as pd

# TODO:  Line 26 indexes
def cstree_borders(cs_tree, x1, y1, x2, y2, contexts, x_p=None, y_p=None, test_fn=None,
                   buffer_range=1, use_energy_sequence=False,
                   weights_range=None, energy_range=None, n_tree=15, verbose=False):
    setting_to_sequence = {}
    setting_to_energy = {}
    setting_to_energy_matrix = {}
    setting_to_energy_sequence = {}
    if weights_range is not None or energy_range is not None:
        if weights_range is None:
            weights_range = (0, 1.0 / (energy_range[1] - energy_range[0]))
        interval = (weights_range[1] - weights_range[0]) / float(n_tree)
        weights = list(np.arange(weights_range[0], weights_range[1], interval)) + [weights_range[1] * 100]
    else:
        weights = [cs_tree.weight]
    trees = {}
    for context in contexts:
        cs_tree.default = context
        cond = pd.Series([False] * len(y1))
        for i in range(buffer_range + 1):
            cond = (cond | (y1.shift(i) == context))
        y1c = y1[cond]
        x1c = x1[cond]
        for weight in weights:
            if verbose:
                print(f'Evaluating: {(context,weight)}')
            setting = (context, weight)
            cs_tree.set_weight(weight)
            cs_tree.fit(x1c, y1c, x_p, y_p)
            trees[(context, weight)] = cs_tree.copy()
            energy = cs_tree.energy(x2, test_fn)
            energy_matrix = [cs_tree.energy(x2[y2 == act], test_fn) for act in contexts]
            predictions = cs_tree.predict(x2)
            setting_to_sequence[setting] = predictions
            setting_to_energy[setting] = energy
            setting_to_energy_matrix[setting] = energy_matrix
            if use_energy_sequence:
                energy_sequence = [cs_tree.energy(x2.iloc[i:i + 1]) for i in range(len(x2))]
                setting_to_energy_sequence[setting] = energy_sequence
    return setting_to_sequence, setting_to_energy, setting_to_energy_matrix, setting_to_energy_sequence, trees


def cstree_weighted(cs_tree, x1, y1, x2, y2, contexts, x_p=None, y_p=None, test_fn=None,
                    weights_range=None, energy_range=None, n_tree=15,
                    use_energy_sequence=False, verbose=False):
    if weights_range is None:
        if energy_range is None:
            raise AssertionError("Either weights_range or energy_range must be set")
        else:
            weights_range = (0, 1.0 / (energy_range[1] - energy_range[0]))
    setting_to_sequence = {}
    setting_to_energy = {}
    setting_to_energy_matrix = {}
    setting_to_energy_sequence = {}
    interval = (weights_range[1] - weights_range[0]) / float(n_tree)
    weights = list(np.arange(weights_range[0], weights_range[1], interval)) + [weights_range[1] * 100]
    trees = {}
    for weight in weights:
        if verbose:
            print(f'Evaluating weight: {weight}')
        cs_tree.set_weight(weight)
        cs_tree.fit(x1, y1, x_p, y_p)
        energy = cs_tree.energy(x2, test_fn)
        energy_matrix = [cs_tree.energy(x2[y2 == act], test_fn) for act in contexts]
        predictions = cs_tree.predict(x2)
        setting_to_sequence[weight] = predictions
        setting_to_energy[weight] = energy
        setting_to_energy_matrix[weight] = energy_matrix
        trees[weight] = cs_tree.copy()
        if use_energy_sequence:
            energy_sequence = [cs_tree.energy(x2.iloc[i:i + 1]) for i in range(len(x2))]
            setting_to_energy_sequence[weight] = energy_sequence
    return setting_to_sequence, setting_to_energy, setting_to_energy_matrix, setting_to_energy_sequence, trees


def general_subsets(x1, y1, x2, y2, classifier, contexts, proportions, setting_to_energy=None, setting_fn_energy=None,
                    subsettings=None, subsetting_to_features=None, n=0, feature_groups=None, setting_fn_features=None,
                    y_p=None, x_p=None, csdt=False, csdt_fn_energy=None, use_energy_sequence=False):
    if setting_to_energy is None and setting_fn_energy is None and not csdt:
        raise AssertionError("Energy parameters missing")
    fill_energy = False
    if setting_to_energy is None:
        fill_energy = True
    parameter_type = 0
    if subsetting_to_features is not None and subsettings is not None:
        parameter_type = 1
    if n != 0 and feature_groups is not None:
        parameter_type = 2
    if n != 0 and setting_fn_features is not None:
        parameter_type = 3
    if parameter_type == 0:
        raise AssertionError("Setting parameters missing")
    if n == 0:
        n = len(subsettings)
    possibilities = [_bin_encode(x, n) for x in range(2 ** n)]
    majority_class = contexts[list(proportions).index(max(proportions))]
    setting_to_sequence = {}
    setting_to_energy_matrix = {}
    setting_to_energy = {}
    setting_to_energy_sequence = {}
    classifiers = {}
    for setting in possibilities:
        features = None
        if parameter_type == 1:
            features = list(set().union(*[subsetting_to_features[key] for (i, key) in enumerate(subsettings)
                                          if setting[i] == 1]))
        if parameter_type == 2:
            features = list(set().union(*[feature_groups[i] for i in range(n) if setting[i] == 1]))
        if parameter_type == 3:
            features = setting_fn_features(setting)
        if len(features) == 0 and not csdt:
            predictions = [majority_class] * len(y2)
        else:
            if x_p is None:
                classifier.fit(x1[features], y1)
            else:
                classifier.fit(x1[features], y1, x_p[features], y_p)
            predictions = classifier.predict(x2[features])
        setting_to_sequence[tuple(setting)] = predictions
        if fill_energy:
            if not csdt:
                setting_to_energy[tuple(setting)] = setting_fn_energy(setting)
            else:
                energy = classifier.energy(x2, csdt_fn_energy)
                energy_matrix = [classifier.energy(x2[y2 == act], csdt_fn_energy) for act in contexts]
                if use_energy_sequence:
                    energy_sequence = [classifier.energy(x2.iloc[i:i + 1]) for i in range(len(x2))]
                    setting_to_energy_sequence[tuple(setting)] = energy_sequence
                setting_to_energy_matrix[tuple(setting)] = energy_matrix
                setting_to_energy[tuple(setting)] = energy
                classifiers[tuple(setting)] = classifier
    return setting_to_sequence, setting_to_energy, setting_to_energy_matrix, setting_to_energy_sequence, classifiers


def _bin_encode(x, length):
    s = str(bin(x))[2:]
    pad = [0] * (length - len(s))
    return pad + [int(x) for x in list(s)]
