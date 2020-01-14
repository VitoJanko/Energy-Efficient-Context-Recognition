from ee import eeutility as util
import ee.scamodel as sca
import ee.dcamodel as dca
import generators as gen
import tester

from random import choice
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle


# TODO: Error handling
# TODO: Double loading
class EnergyOptimizer:

    def __init__(self, sequence=None, contexts=None, setting_to_energy=None, setting_to_sequence=None,
                 quality_metric=None, sensors_off=None, sensors_on=None,
                 path=None):
        self.proportions = None
        self.sequence = None
        self.setting_to_confusion = None
        self.transitions = None
        self.avg_lengths = None
        self.sensors_off = 0 if sensors_off is None else sensors_off
        self.sensors_on = 1 if sensors_on is None else sensors_on
        self.setting_to_energy = None
        self.setting_to_sequence = None
        self.quality_metric = util.accuracy_matrix
        self.setting_to_sequence = None
        self.path = "./"
        self.contexts = contexts
        self.setting_to_energy_matrix = None
        self.setting_to_energy_sequence = None
        self.settings = None
        if path is not None:
            self.path = path
        if quality_metric is not None:
            self.quality_metric = quality_metric
        if sequence is not None:
            self.sequence = pd.Series(sequence).reset_index(drop=True)
            self._calc_contexts()
            self._calc_transitions()
        if setting_to_sequence is not None:
            self.set_settings(setting_to_sequence, setting_to_energy)

    def set_sequence(self, sequence):
        self.sequence = pd.Series(sequence).reset_index(drop=True)
        self._calc_contexts()
        self._calc_transitions()

    def set_path(self, path):
        self.path = path

    def set_settings(self, setting_to_sequence, setting_to_energy=None, setting_fn_energy=None):
        if setting_to_energy is None and setting_fn_energy is None:
            raise AssertionError("Either setting_to_energy or setting_fn_energy must not be None")
        self.setting_to_energy = setting_to_energy
        if setting_to_sequence is not None:
            self.setting_to_sequence = {k: pd.Series(v) for (k, v) in setting_to_sequence.items()}
            self.settings = list(setting_to_sequence.keys())
            if setting_fn_energy is not None:
                self.setting_to_energy = self._calc_energy_fromfn(setting_fn_energy)
            self._calc_confusions()

    def set_dca_costs(self, cost_off, cost_on):
        self.sensors_on = cost_on
        self.sensors_off = cost_off

    def _calc_contexts(self):
        if self.contexts is None:
            self.contexts = sorted(self.sequence.unique())

    def _calc_transitions(self):
        self.transitions = util.get_transitions(self.sequence, self.contexts)
        self.proportions = util.get_steady_state(self.transitions)
        self.avg_lengths = util.average_length(self.transitions)

    def _calc_confusions(self):
        self.setting_to_confusion = {
            setting: util.normalize_matrix(confusion_matrix(self.sequence, pred, self.contexts), rows=True)
            for setting, pred in self.setting_to_sequence.items()}

    def _calc_energy_fromfn(self, setting_fn_energy):
        return {setting: setting_fn_energy(setting) for setting in self.settings}

    def _initialize_settings(self):
        if self.setting_to_energy is None:
            self.setting_to_energy = {}
        if self.setting_to_sequence is None:
            self.setting_to_sequence = {}
        if self.setting_to_energy_matrix is None:
            self.setting_to_energy_matrix = {}
        if self.setting_to_energy_sequence is None:
            self.setting_to_energy_sequence = {}

    def _update_settings(self, setting_to_sequence, setting_to_energy, setting_to_energy_matrix,
                         setting_to_energy_sequence, name):
        self.setting_to_sequence.update(setting_to_sequence)
        self.setting_to_energy_sequence.update(setting_to_energy_sequence)
        self.setting_to_energy.update(setting_to_energy)
        self.setting_to_energy_matrix.update(setting_to_energy_matrix)
        self.set_settings(self.setting_to_sequence, self.setting_to_energy)
        self.save_config(name)

    def get_quality(self):
        return {s: self.quality_metric((self.setting_to_confusion[s].T * self.proportions).T) for s in self.settings}

    def get_energy_quality(self):
        return {s: (self.quality_metric((self.setting_to_confusion[s].T * self.proportions).T),
                    self.setting_to_energy[s])
                for s in self.settings}

    def get_summary(self):
        summary = pd.DataFrame()
        summary["Proportions"] = pd.Series(self.proportions, self.contexts)
        summary["Average lengths"] = pd.Series(self.avg_lengths, self.contexts)
        print(summary)

    def load_data_config(self, data_name, config_name):
        self.load_data(data_name)
        self.load_config(config_name)

    def load_data(self, name):
        dbfile = open(self.path + "/" + name, 'rb')
        sequence = pickle.load(dbfile)
        dbfile.close()
        self.set_sequence(sequence)

    def load_config(self, name):
        dbfile = open(self.path + "/" + name, 'rb')
        loaded = pickle.load(dbfile)
        self.settings, self.setting_to_sequence, self.setting_to_energy, self.setting_to_confusion = loaded[0:4]
        if len(loaded) > 4:
            self.setting_to_energy_matrix = loaded[4]
            self.setting_to_energy_sequence = loaded[5]
        dbfile.close()

    def load_solution(self, name):
        dbfile = open(self.path + "/" + name, 'rb')
        loaded = pickle.load(dbfile)
        hof, solutions = loaded
        dbfile.close()
        return hof, solutions

    def save_data(self, name):
        if name is not None:
            dbfile = open(self.path + "/" + name, 'wb')
            pickle.dump(self.sequence, dbfile)
            dbfile.close()

    def save_solution(self, hof, solution, name):
        if name is not None:
            dbfile = open(self.path + "/" + name, 'wb')
            pickle.dump((hof, solution), dbfile)
            dbfile.close()

    def save_config(self, name):
        if name is not None:
            dbfile = open(self.path + "/" + name, 'wb')
            pickle.dump((self.settings, self.setting_to_sequence, self.setting_to_energy,
                         self.setting_to_confusion, self.setting_to_energy_matrix,
                         self.setting_to_energy_sequence),
                        dbfile)
            dbfile.close()

    def _encrypt_hof(self, hof):
        return [[self.settings[sett] for sett in assignment] for assignment in hof]

    def _decrypt_hof(self, hof):
        return [[self.settings.index(sett) for sett in assignment] for assignment in hof]

    def _encrypt_solution(self, solution):
        return [self.settings[sett] for sett in solution]

    def _decrypt_solution(self, solution):
        return [self.settings.index(sett) for sett in solution]

    def sca_real(self, tradeoffs, name=None,  cstree_energy=False):
        tradeoffs = self._wrap(tradeoffs)
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        solutions = [self._sca_real_single(configuration, sequence_true, cstree_energy) for configuration in tradeoffs]
        self.save_solution(tradeoffs, solutions, name)
        return solutions

    def _sca_real_single(self, configuration, sequence_true, cstree_energy=False):
        sequences = [self.setting_to_sequence[s].apply(lambda x: self.contexts.index(x)) for s in configuration]
        energy_costs = [self.setting_to_energy[s] for s in configuration]
        energy_sequences = None
        if cstree_energy:
            energy_sequences = [self.setting_to_energy_sequence[s] for s in configuration]
        return tester.test_sca_dca(sequence_true, sequences, [1] * len(self.contexts), energy_costs,
                                   self.sensors_off, self.quality_metric, 1, energy_sequences)

    def sca_simple(self, tradeoffs, name=None):
        tradeoffs = self._wrap(tradeoffs)
        solutions = [self._sca_simple_single(configuration) for configuration in tradeoffs]
        self.save_solution(tradeoffs, solutions, name)
        return solutions

    def _sca_simple_single(self, configuration):
        self._checks(configuration)
        confusions = [self.setting_to_confusion[setting] for setting in configuration]
        energies = [self.setting_to_energy[setting] for setting in configuration]
        return sca.sca_simple_evaluation(self.proportions, confusions, energies, self.quality_metric)

    def sca_model(self, tradeoffs, encrypted=False, name=None, cstree=False):
        tradeoffs = self._wrap(tradeoffs)
        if encrypted:
            tradeoffs = self._encrypt_hof(tradeoffs)
        solutions = [self._sca_model_single(configuration, cstree=cstree) for configuration in tradeoffs]
        self.save_solution(tradeoffs, solutions, name)
        return solutions

    def _sca_model_single(self, configuration, return_cf=False, cstree=False):
        self._checks(configuration)
        confusions = [self.setting_to_confusion[setting] for setting in configuration]
        energies = [self.setting_to_energy[setting] for setting in configuration]
        if cstree:
            energies = [self.setting_to_energy_matrix[setting] for setting in configuration]
        quality = (lambda x: x) if return_cf else self.quality_metric
        return sca.sca_evaluation(self.transitions, confusions, energies, quality)

    def find_sca_tradeoffs(self, binary_representation=False, name=None, cstree=False):
        if self.setting_to_confusion is None or self.setting_to_energy is None:
            raise AssertionError("Confusion matrices or energy estimations missing!")
        if not binary_representation:
            tradeoffs = sca.sca_find_tradeoffs("enumerate", len(self.settings), len(self.contexts),
                                               self, NGEN=200, cstree=cstree)
            tradeoffs = self._encrypt_hof(tradeoffs)
        else:
            tradeoffs = sca.sca_find_tradeoffs("binary", len(self.settings[0]), len(self.contexts),
                                               self, NGEN=200, cstree=cstree)
        values = self.sca_model(tradeoffs)
        self.save_solution(tradeoffs, values, name)
        return tradeoffs, values

    def find_sca_static(self, name=None):
        if self.setting_to_sequence is None or self.setting_to_energy is None:
            raise AssertionError("Settings are not set")
        points = []
        for s in self.settings:
            cf = confusion_matrix(self.sequence, self.setting_to_sequence[s], self.contexts)
            e = self.setting_to_energy[s]
            a = self.quality_metric(cf)
            points.append((1 - a, e, s))
        pareto_points = util.pareto_dominance(points)
        pareto_hof = list(zip(*pareto_points))[2]
        pareto_sols = util.reverse_first(zip(*list(zip(*pareto_points))[0:2]))
        self.save_solution(pareto_hof, pareto_sols, name)
        return pareto_hof, pareto_sols

    def find_sca_random(self, n_samples=100):
        configs = []
        for i in range(n_samples):
            configs.append([choice(self.settings) for _ in range(len(self.contexts))])
        return configs

    @staticmethod
    def _wrap(tradeoffs):
        if type(tradeoffs[0]) != list or type(tradeoffs) == tuple:
            tradeoffs = [tradeoffs]
        return tradeoffs

    def _checks(self, configuration):
        if self.setting_to_confusion is None or self.setting_to_energy is None:
            raise AssertionError("Confusion matrices or energy estimations missing!")
        if len(configuration) != len(self.contexts):
            raise ValueError("Assignment length does not match the number of contexts!")

    def dca_real(self, tradeoffs, setting=None, active=1, name=None):
        tradeoffs = self._wrap(tradeoffs)
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        if setting is not None:
            sequence_predicted = self.setting_to_sequence[setting].apply(lambda x: self.contexts.index(x))
            energy = self.setting_to_energy[setting]
        else:
            sequence_predicted = sequence_true
            energy = self.sensors_on
        solutions = [self._dca_real_single(configuration, sequence_true, sequence_predicted, energy, active)
                     for configuration in tradeoffs]
        self.save_solution(tradeoffs, solutions, name)
        return solutions

    def _dca_real_single(self, configuration, sequence_true, sequence_predicted, energy, active):
        return tester.test_sca_dca(sequence_true, [sequence_predicted] * len(self.contexts), configuration,
                                   [energy] * len(self.contexts), self.sensors_off, self.quality_metric, active)

    def dca_model(self, tradeoffs, active=1, setting=None, cf=None, max_cycle=None, energy_costs=None, name=None):
        tradeoffs = self._wrap(tradeoffs)
        cf = self._calc_duty_cf(setting, cf)
        if max_cycle is None:
            max_cycle = max([max(config) for config in tradeoffs])
        prob, sleeping, working = dca.precalculate(self.transitions, max_cycle + 1, active)
        if energy_costs is None:
            energy_costs = self.sensors_on
            if (setting is not None) and (self.setting_to_energy is not None):
                energy_costs = self.setting_to_energy[setting]
        solutions = [dca.dca_evaluation(self.transitions, configuration, self.quality_metric, active=active,
                                        sleeping_exp=sleeping, working_exp=working, prob=prob, confusion=cf,
                                        energy_off=self.sensors_off, energy_costs=energy_costs)
                     for configuration in tradeoffs]
        self.save_solution(tradeoffs, solutions, name)
        return solutions

    def find_dca_static(self, max_cycle, name=None, setting=None, active=1):
        tradeoffs = [(i,)*len(self.contexts) for i in range(1, max_cycle)]
        sols = self.dca_model(tradeoffs, setting=setting, active=active, max_cycle=max_cycle)
        self.save_solution(tradeoffs, sols, name)
        return tradeoffs, sols

    def find_dca_tradeoffs(self, max_cycle=10, active=1, seeded=True, setting=None, cf=None, name=None,
                           energy_costs=None, ngen=200):
        cf = self._calc_duty_cf(setting, cf)
        prob, sleeping, working = dca.precalculate(self.transitions, max_cycle + 1, active)
        mx = self.sensors_on
        if (setting is not None) and (self.setting_to_energy is not None):
            mx = self.setting_to_energy[setting]

        def evl(x):
            dca.dca_evaluation(transitions=self.transitions, lengths=x, active=active, confusion=cf,
                               evaluator=self.quality_metric, prob=prob, sleeping_exp=sleeping,
                               working_exp=working, energy_off=self.sensors_off, energy_costs=mx)
        if energy_costs is not None:
            def evl(x):
                dca.dca_evaluation(transitions=self.transitions, lengths=x, active=active, confusion=cf,
                                   evaluator=self.quality_metric, prob=prob, sleeping_exp=sleeping,
                                   working_exp=working, energy_off=self.sensors_off,
                                   energy_costs=energy_costs)

        tradeoffs = dca.dca_find_tradeoffs(len(self.contexts), max_cycle, evl, seeded=seeded, NGEN=ngen)
        solutions = [evl(h) for h in tradeoffs]
        hof = [list(h) for h in tradeoffs]
        self.save_solution(hof, solutions, name)
        return hof, solutions

    def find_dca_random(self, n_samples=100, max_cycles=10):
        configs = []
        for i in range(n_samples):
            configs.append([choice(range(1, max_cycles)) for _ in range(len(self.contexts))])
        return configs

    def _calc_duty_cf(self, setting, cf):
        if (cf is None) and (setting is not None) and (self.setting_to_confusion is not None):
            cf = self.setting_to_confusion[setting]
        if cf is not None:
            cf = util.normalize_matrix(cf, rows=True)
        return cf

    def sca_dca_real(self, tradeoffs, active=1, name=None, cstree_energy=False):
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        sols = [self._sca_dca_real_single(config, active, sequence_true, cstree_energy) for config in tradeoffs]
        self.save_solution(tradeoffs, sols, name)
        return sols

    def _sca_dca_real_single(self, config, active, sequence_true, cstree_energy=None):
        if type(config) == tuple:
            energy_costs = [self.setting_to_energy[s] for s in config[0]]
            sequences = [self.setting_to_sequence[s].apply(lambda x: self.contexts.index(x)) for s in config[0]]
            energy_sequences = None
            if cstree_energy:
                energy_sequences = [self.setting_to_energy_sequence[s] for s in config[0]]
            return tester.test_sca_dca(sequence_true, sequences, config[1], energy_costs,
                                       self.sensors_off, self.quality_metric, active, energy_sequences)
        else:
            return self.sca_real(config)[0]

    # TODO: Add energy costs for csdt and sca/dca
    # TODO: Add solution types
    # TODO: wrapping type

    def sca_dca_model(self, tradeoffs, active=1, cstree=False, name=None):
        sols = [self._sca_dca_model_single(h, active=active, cstree=cstree) for h in tradeoffs]
        self.save_solution(tradeoffs, sols, name)
        return sols

    def _sca_dca_model_single(self, setting, active=1, cstree=False):
        if type(setting) == tuple:
            cf = self._sca_model_single(setting[0], return_cf=True)[0]
            energy_costs = [self.setting_to_energy[s] for s in setting[0]]
            if cstree:
                energy_costs = [self.setting_to_energy_matrix[row] for row in setting[0]]
            return self.dca_model(setting[1], cf=cf, active=active, energy_costs=energy_costs)[0]
        else:
            return self._sca_model_single(setting, cstree=cstree)

    def find_sca_dca_tradeoffs(self, hof=None, solutions=None, dca_indices=None, n_points=5,
                               binary_representation=False, name=None, max_cycle=10, active=1,
                               verbose=False, cstree=False):
        if solutions is None:
            if verbose:
                print("Calculating SCA trade-offs...")
            hof, solutions = self.find_sca_tradeoffs(binary_representation)
        if dca_indices is None:
            if verbose:
                print("Searching for suitable DCA starting points...")
            start_tradeoffs, start_points = self._calc_sca_dca_points(hof, solutions, n_points)
            if verbose:
                print("Points found: " + str(len(start_points)))
        else:
            start_tradeoffs = [hof[i] for i in dca_indices]
            start_points = [solutions[i] for i in dca_indices]
        all_points = []
        pareto_points = [(1 - a, e, h) for h, (a, e) in zip(hof, solutions)]
        for (h, point) in zip(start_tradeoffs, start_points):
            if verbose:
                print("Expanding trade-off: " + str(point))
            energy_costs = [self.setting_to_energy[setting] for setting in h]
            if cstree:
                energy_costs = [self.setting_to_energy_matrix[setting] for setting in h]
            cf = self._sca_model_single(h, return_cf=True)[0]
            dca_hof, dca_s = self.find_dca_tradeoffs(max_cycle=max_cycle, cf=cf, active=active,
                                                     energy_costs=energy_costs)
            sca_dca_hof = [(h, dca_h) for dca_h in dca_hof]
            all_points.append((sca_dca_hof, dca_s))
            pareto_points += [(1 - a, e, sca_dca_h) for (a, e), sca_dca_h in zip(dca_s, sca_dca_hof)]
        pareto_points = util.pareto_dominance(pareto_points)
        pareto_hof = list(zip(*pareto_points))[2]
        pareto_sols = util.reverse_first(list(zip(*list(zip(*pareto_points))[0:2])))
        self.save_solution(pareto_hof, pareto_sols, name)
        return pareto_hof, pareto_sols

    # TODO: Make more accurate
    @staticmethod
    def _calc_sca_dca_points(hof, solutions, number=5):
        if number > len(hof):
            pass
        acc_threshold = (solutions[0][0] - solutions[-1][0]) / (2.0 * number)
        energy_threshold = (solutions[0][1] - solutions[-1][1]) / (2.0 * number)
        while True:
            points, hofs, x = [], [], solutions[0][0] + 1
            acc_lst, energy_lst = zip(*solutions)
            for i, (acc, energy, h) in enumerate(zip(acc_lst, energy_lst, hof)):
                if acc < x and (len(points) == 0 or energy < points[-1][1] - energy_threshold):
                    x - acc_threshold
                    points.append((acc, energy))
                    hofs.append(h)
            if len(points) > number:
                return hofs, points
            else:
                acc_threshold /= 2
                energy_threshold /= 2

    def add_subsets(self, x1, y1, x2, y2, classifier, setting_to_energy=None, setting_fn_energy=None, subsettings=None,
                    subsetting_to_features=None, n=0, feature_groups=None, setting_fn_features=None, name=None,
                    y_p=None, x_p=None, csdt=False, csdt_fn_energy=None, use_energy_sequence=False):
        config = gen.general_subsets(x1, y1, x2, y2, classifier, self.contexts, self.proportions, setting_to_energy,
                                     setting_fn_energy, subsettings, subsetting_to_features, n, feature_groups,
                                     setting_fn_features, y_p, x_p, csdt, csdt_fn_energy, use_energy_sequence)
        self._initialize_settings()
        setting_to_sequence, setting_to_energy, setting_to_energy_matrix, \
            setting_to_energy_sequence, classifiers = config
        self._update_settings(setting_to_sequence, setting_to_energy, setting_to_energy_matrix,
                              setting_to_energy_sequence, name)
        return classifiers

    def add_csdt_weighted(self, cs_tree, x1, y1, x2, y2, x_p=None, y_p=None, test_fn=None, verbose=True,
                          weights_range=None, energy_range=None, n_tree=15, name=None, use_energy_sequence=False):
        config = gen.cstree_weighted(cs_tree, x1, y1, x2, y2, self.contexts, x_p, y_p, test_fn, weights_range,
                                     energy_range, n_tree, use_energy_sequence, verbose)
        self._initialize_settings()
        setting_to_sequence, setting_to_energy, setting_to_energy_matrix, setting_to_energy_sequence, trees = config
        self._update_settings(setting_to_sequence, setting_to_energy, setting_to_energy_matrix,
                              setting_to_energy_sequence, name)
        return trees

    def add_csdt_borders(self, cs_tree, x1, y1, x2, y2, x_p=None, y_p=None, test_fn=None,
                         buffer_range=1, name=None, use_energy_sequence=False,
                         weights_range=None, energy_range=None, n_tree=15):
        config = gen.cstree_borders(cs_tree, x1, y1, x2, y2, self.contexts, x_p, y_p, test_fn,
                                    buffer_range, use_energy_sequence,
                                    weights_range, energy_range, n_tree)
        self._initialize_settings()
        setting_to_sequence, setting_to_energy, setting_to_energy_matrix, setting_to_energy_sequence, trees = config
        self._update_settings(setting_to_sequence, setting_to_energy, setting_to_energy_matrix,
                              setting_to_energy_sequence, name)
        return trees
