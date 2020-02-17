from eecr import eeutility as util
import eecr.scamodel as sca
import eecr.dcamodel as dca
import eecr.generators as gen
import eecr.tester as test
import eecr.altmethods as alt

from random import choice
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
import os


# TODO: Error handling
# TODO: Double loading
# TODO: Related work
class EnergyOptimizer:

    #A class designed for optimizing the energy-efficiency of a context recognition system.

    #A complex description to be written later.



    def __init__(self, sequence=None, contexts=None, setting_to_energy=None, setting_to_sequence=None,
                 quality_metric=None, sensors_off=None, sensors_on=None,
                 path=None):
        """

        :param sequence: a sequence of contexts to be used for testing the energy-efficiency of
                         settings. Their ordering and proportions should be representative for
                         the domain
        :param contexts: the list of contexts in the domain. If not provided, they will be inferred as
                         all unique elements of the ``sequence`` parameter in the alphabetical order
        :param setting_to_energy: a dictionary, where the keys are the possible settings and the
                                   values are the energy costs (per time unit) when using that setting
        :param setting_to_sequence:  a dictionary, where the keys are the possible settings and the
                                    values are lists of contexts. Each list represents the original
                                    sequence (see :func:`~EnergyOptimizer.set_sequence`),
                                    classified using the corresponding setting
        :param quality_metric: a function that maps a confusion matrix to a quality indicator.
                               Accuracy score is used by default
        :param sensors_off: the energy cost per time unit when the system is sleeping
        :param sensors_on:  the energy cost per time unit when the system is working
        :param path: the root of the path used for saving and loading objects (e.g. with
                     :func:`~EnergyOptimizer.save_data` or :func:`~EnergyOptimizer.load_data`)
        """
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
        """
        Sets the sequence of contexts that will be used for testing the energy-efficiency of settings.

        :param sequence: a sequence of contexts. Their ordering and proportions should be
                         representative for the domain
        """
        self.sequence = pd.Series(sequence).reset_index(drop=True)
        self._calc_contexts()
        self._calc_transitions()

    def set_path(self, path):
        """
        Sets the path to a folder from which the data is loaded and to which data is saved.

        :param path: relative path to a folder
        """
        self.path = path

    def set_settings(self, setting_to_sequence, setting_to_energy=None, setting_fn_energy=None):
        """
        Defines the possible settings and their performance/energy for the optimization.

        Either ``setting_to_energy`` or ``setting_fn_energy parameter`` must be provided. Providing both or
        neither may result in unexpected behavior.

        :param setting_to_sequence: a dictionary, where the keys are the possible settings and the
                                    values are lists of contexts. Each list represents the original
                                    sequence (see :func:`~EnergyOptimizer.set_sequence`),
                                    classified using the corresponding setting
        :param setting_to_energy:  a dictionary, where the keys are the possible settings and the
                                   values are the energy costs (per time unit) when using that setting
        :param setting_fn_energy: a function to be used if the ``setting_to_energy`` is None.
                                  This function should take a setting as the input and return energy
                                  costs (per time unit) when using that setting
        """
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
        """
        Sets the base energy costs for duty-cycling.

        This costs will be used for all DCA methods (e.g. :func:`~EnergyOptimizer.dca_model`,
        :func:`~EnergyOptimizer.dca_real`, :func:`~EnergyOptimizer.find_dca_tradeoffs` etc.).
        This overrides the default costs of 1 when the system is working and 0 when the system
        is sleeping.

        :param cost_off: the energy cost per time unit when the system is sleeping
        :param cost_on: the energy cost per time unit when the system is working.
                        If a setting is specified when using any of the DCA methods
                        this value will be ignored and the energy cost of that setting
                        will be used instead
        """
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

    # TODO: This is a slightly inaccurate way to report accuracy
    def quality(self):
        """
        Returns a summary of all settings and their classification performances.

        :return: a dictionary, where the keys are the possible settings and values the classification
                 performances according the specified quality metric
        """
        return {s: self.quality_metric((self.setting_to_confusion[s].T * self.proportions).T) for s in self.settings}

    def energy_quality(self):
        """
        Returns a summary of all settings and their classification/energy performances.

        :return: a dictionary, where the keys are the possible settings and values are (q,e) tuples,
                 where the "q" represents the  classification performance according to the specified
                 quality metric and "e" represents the energy cost of that setting.
        """
        return {s: (self.quality_metric((self.setting_to_confusion[s].T * self.proportions).T),
                    self.setting_to_energy[s])
                for s in self.settings}

    def summary(self):
        """
        Prints a summary (mathematical properties) of the dataset.
        """
        summa = pd.DataFrame()
        summa["Proportions"] = pd.Series(self.proportions, self.contexts)
        summa["Average lengths"] = pd.Series(self.avg_lengths, self.contexts)
        print(summa)

    def load_data_config(self, data_name=None, config_name=None, sample_dataset=None):
        """
        Loads the sequence and settings information from files.

        This is equivalent to sequentially calling the :func:`~EnergyOptimizer.load_data` and
        :func:`~EnergyOptimizer.load_config` functions.

        :param data_name: relative path to a file with the sequence
        :param config_name: relative path to a file with the settings information
        :param sample_dataset: if specified, a sample dataset will be loaded (currently only "SHL"
                               keyword works)
        """
        if sample_dataset is not None:
            if sample_dataset == "SHL":
                temp = self.path
                self.path = os.path.join(os.path.dirname(__file__), './Datasets')
                self.load_data_config("SHL_data", "SHL_config")
                self.path = test
        else:
            self.load_data(data_name)
            self.load_config(config_name)

    def load_data(self, name):
        """
        Loads the base sequence from a file (instead of calling :func:`~EnergyOptimizer.set_sequence`).

        :param name: relative path to a file
        """
        dbfile = open(self.path + "/" + name, 'rb')
        sequence = pickle.load(dbfile)
        dbfile.close()
        self.set_sequence(sequence)

    def load_config(self, name):
        """
        Loads the settings information from a file (instead of calling :func:`~EnergyOptimizer.set_settings`)

        :param name: relative path to a file
        """
        dbfile = open(self.path + "/" + name, 'rb')
        loaded = pickle.load(dbfile)
        self.settings, self.setting_to_sequence, self.setting_to_energy, self.setting_to_confusion = loaded[0:4]
        if len(loaded) > 4:
            self.setting_to_energy_matrix = loaded[4]
            self.setting_to_energy_sequence = loaded[5]
        dbfile.close()

    def load_solution(self, name):
        """
        Loads a set of energy-efficient solutions from a file.

        :param name: relative path to a file
        :return: two lists representing different tradeoffs. First list contains the configurations,
                 while the other their evaluations. Evaluations are represented by a tuple with the
                 form (quality, energy)
        """
        dbfile = open(self.path + "/" + name, 'rb')
        loaded = pickle.load(dbfile)
        hof, solutions = loaded
        dbfile.close()
        return hof, solutions

    def save_data(self, name):
        """
        Saves the base sequence set with :func:`~EnergyOptimizer.set_settings` to a file.

        :param name: relative path to a file
        """
        if name is not None:
            dbfile = open(self.path + "/" + name, 'wb')
            pickle.dump(self.sequence, dbfile)
            dbfile.close()

    def save_solution(self, configurations, values, name):
        """
        Saves a set of energy-efficient solutions from a file.

        :param configurations: a list of configurations
        :param values: a list of configuration evaluation in form of (quality, energy)
        :param name: relative path to a file
        """
        if name is not None:
            dbfile = open(self.path + "/" + name, 'wb')
            pickle.dump((configurations, values), dbfile)
            dbfile.close()

    def save_config(self, name):
        """
        Save the settings information (set with :func:`~EnergyOptimizer.set_settings`) to a file.

        :param name: relative path to a file
        """
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

    def sca_real(self, configurations, name=None,  cstree_energy=False):
        """
        Tests the SCA configurations using a simulation.

        This method reads from different sequences (that correspond to different settings), switching
        between them as context changes. This simulates a real-life environment where contexts are
        classified one-by-one using different system settings. This is slower than using the
        mathematical model instead ( :func:`~EnergyOptimizer.sca_model`).

        :param configurations: a list of SCA configurations. A configuration is a list of the same
                               length as the number of contexts, each element being a setting.
                               Optionally this parameter can be a single configuration instead of a
                               list
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :param cstree_energy: this flag slightly increases the accuracy of the simulation when using
                              the cost-sensitive decision trees. In order to use it, the energy
                              sequence  must be precalculated, by using ``cstree_energy`` flag when
                              generating the trees (e.g. in
                              :func:`~EnergyOptimizer.add_csdt_weighted` or
                              :func:`~EnergyOptimizer.add_csdt_borders`)
        :return: a list of trade-offs, in the form (quality, energy). i-th element represents the
                 evaluation of the i-th configuration
        """
        configurations = self._wrap(configurations)
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        tradeoffs = [self._sca_real_single(configuration, sequence_true, cstree_energy) for configuration in configurations]
        self.save_solution(configurations, tradeoffs, name)
        return tradeoffs

    def _sca_real_single(self, configuration, sequence_true, cstree_energy=False):
        sequences = [self.setting_to_sequence[s].apply(lambda x: self.contexts.index(x)) for s in configuration]
        energy_costs = [self.setting_to_energy[s] for s in configuration]
        energy_sequences = None
        if cstree_energy:
            energy_sequences = [self.setting_to_energy_sequence[s] for s in configuration]
        return test.test_sca_dca(sequence_true, sequences, [1] * len(self.contexts), energy_costs,
                                   self.sensors_off, self.quality_metric, 1, energy_sequences)

    def sca_simple(self, configurations, name=None):
        """
        Tests the SCA configurations using a simple mathematical model.

        The model used simply multiplies the performance of each setting with the expected proportion
        of time that setting is in use. E.g if configuration consists of two settings, one with
        accuracy 60% and other with 100%, and if both contexts appear equally often, then the expected
        accuracy is 80%.

        :param configurations: a list of SCA configurations. A configuration is a list of the same
                               length as the number of contexts, each element being a setting.
                               Optionally this parameter can be a single configuration instead of a
                               list
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :return: a list of trade-offs, in the form (quality, energy). i-th element represents the
                 evaluation of the i-th configuration
        """
        configurations = self._wrap(configurations)
        tradeoffs = [self._sca_simple_single(configuration) for configuration in configurations]
        self.save_solution(configurations, tradeoffs, name)
        return tradeoffs

    def _sca_simple_single(self, configuration):
        self._checks(configuration)
        confusions = [self.setting_to_confusion[setting] for setting in configuration]
        energies = [self.setting_to_energy[setting] for setting in configuration]
        return sca.sca_simple_evaluation(self.proportions, confusions, energies, self.quality_metric)

    def sca_model(self, configurations, name=None, cstree=False, encrypted=False):
        """
        Tests the SCA configurations using the SCA mathematical model.

        :param configurations: a list of SCA configurations. A configuration is a list of the same
                               length as the number of contexts, each element being a setting.
                               Optionally this parameter can be a single configuration instead of a
                               list
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :param cstree: a flag indicating that cost-sensitive trees are used. Makes the evaluation
                       more accurate if used
        :param encrypted: a flag for internal use (do not change its value)
        :return: a list of trade-offs, in the form (quality, energy). i-th element represents the
                 evaluation of the i-th configuration
        """
        configurations = self._wrap(configurations)
        if encrypted:
            configurations = self._encrypt_hof(configurations)
        tradeoffs = [self._sca_model_single(configuration, cstree=cstree) for configuration in configurations]
        self.save_solution(configurations, tradeoffs, name)
        return tradeoffs

    def _sca_model_single(self, configuration, return_cf=False, cstree=False):
        self._checks(configuration)
        confusions = [self.setting_to_confusion[setting] for setting in configuration]
        energies = [self.setting_to_energy[setting] for setting in configuration]
        if cstree:
            energies = [self.setting_to_energy_matrix[setting] for setting in configuration]
        quality = (lambda x: x) if return_cf else self.quality_metric
        return sca.sca_evaluation(self.transitions, confusions, energies, quality, matrix_energy=cstree)

    def find_sca_tradeoffs(self, binary_representation=False, name=None, cstree=False):
        """
        Attempts to find best SCA trade-offs for the current dataset.

        Uses the NSGA-II algorithm to search the space of different configurations and finds and
        returns the Pareto-optimal ones.

        :param binary_representation: a flag indicating that settings are represented by a binary
                                      list. In this case, a different mutation/crossover will be used
                                      for the NSGA-II algorithm
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :param cstree: a flag indicating that cost-sensitive trees are used. Makes the evaluation
                       more accurate if used
        :return: two lists, the first contains pareto-optimal configurations, the second their
                 evaluations in the form (quality, energy).
        """
        if self.setting_to_confusion is None or self.setting_to_energy is None:
            raise AssertionError("Confusion matrices or energy estimations missing!")
        if not binary_representation:
            tradeoffs = sca.sca_find_tradeoffs("enumerate", len(self.settings), len(self.contexts),
                                               self, NGEN=200, cstree=cstree)
            tradeoffs = self._encrypt_hof(tradeoffs)
        else:
            tradeoffs = sca.sca_find_tradeoffs("binary", len(self.settings[0]), len(self.contexts),
                                               self, NGEN=200, cstree=cstree)
        values = self.sca_model(tradeoffs, cstree=cstree)
        self.save_solution(tradeoffs, values, name)
        return tradeoffs, values

    def find_sca_static(self, name=None):
        """
       Returns all Pareto-optimal configurations where the same setting is used for all contexts.

       :param name: None or relative path to a file. In latter case the path will be used to save
                    the generated configurations and trade-offs

       :return: two lists, the first contains pareto-optimal configurations, the second their
                evaluations in the form (quality, energy).
       """
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
        """
        Returns ``n_samples`` random SCA configurations.

        :param n_samples: number of configurations to return
        :return: a list of configurations
        """
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

    def dca_real(self, configurations, setting=None, active=1, name=None):
        """
        Tests the DCA configurations using a simulation.

        This method reads from a sequence (either base one or one that corresponds to the given
        setting), skipping some in a way that simulates duty-cycling in a real-life environment.
        This is slower than using the mathematical model ( :func:`~EnergyOptimizer.dca_model`).

        :param configurations: a list of DCA configurations. A configuration is a list of the same
                               length as the number of contexts, each element being an integer >= 1.
                               Optionally this parameter can be a single configuration instead of a
                               list
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :param setting: the setting that is in use while duty-cycling. If None is provided the
                        function assumes that the classification accuracy is 100% and the energy
                        cost is 1
        :param active: the length of the active period. For most purposes this length should be left
                       as the default of 1
        :return: a list of trade-offs, in the form (quality, energy). i-th element represents the
                 evaluation of the i-th configuration
        """
        configurations = self._wrap(configurations)
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        if setting is not None:
            sequence_predicted = self.setting_to_sequence[setting].apply(lambda x: self.contexts.index(x))
            energy = self.setting_to_energy[setting]
        else:
            sequence_predicted = sequence_true
            energy = self.sensors_on
        tradeoffs = [self._dca_real_single(configuration, sequence_true, sequence_predicted, energy, active)
                     for configuration in configurations]
        self.save_solution(configurations, tradeoffs, name)
        return tradeoffs

    def _dca_real_single(self, configuration, sequence_true, sequence_predicted, energy, active):
        return test.test_sca_dca(sequence_true, [sequence_predicted] * len(self.contexts), configuration,
                                   [energy] * len(self.contexts), self.sensors_off, self.quality_metric, active)

    def dca_model(self, configurations, active=1, setting=None, cf=None, max_cycle=None,
                  energy_costs=None, name=None):
        """
        Tests the DCA configurations using the DCA mathematical model.

        :param configurations: a list of DCA configurations. A configuration is a list of the same
                               length as the number of contexts, each element being an integer >= 1.
                               Optionally this parameter can be a single configuration instead of a
                               list
        :param active: the length of the active period. For most purposes this length should be left
                       as the default of 1
        :param setting: the setting that is in use while duty-cycling. If None is provided the
                        function assumes that the classification accuracy is 100% and the energy
                        cost is 1
        :param cf: uses the given confusion matrix (must be normalized, so each row sums to 1)
                   instead of the one prescribed by the ``setting`` parameter
        :param max_cycle: the maximum duty-cycle length out of any found in configurations. If set
                          the evaluation time will be slightly faster
        :param energy_costs: internal parameter (its value should not be changed)
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :return: a list of trade-offs, in the form (quality, energy). i-th element represents the
                 evaluation of the i-th configuration
        """
        configurations = self._wrap(configurations)
        cf = self._calc_duty_cf(setting, cf)
        if max_cycle is None:
            max_cycle = max([max(config) for config in configurations])
        prob, sleeping, working = dca.precalculate(self.transitions, max_cycle + 1, active)
        if energy_costs is None:
            energy_costs = self.sensors_on
            if (setting is not None) and (self.setting_to_energy is not None):
                energy_costs = self.setting_to_energy[setting]
        tradeoffs = [dca.dca_evaluation(self.transitions, configuration, self.quality_metric, active=active,
                                        sleeping_exp=sleeping, working_exp=working, prob=prob, confusion=cf,
                                        energy_off=self.sensors_off, energy_costs=energy_costs)
                     for configuration in configurations]
        self.save_solution(configurations, tradeoffs, name)
        return tradeoffs

    def find_dca_static(self, max_cycle, name=None, setting=None, active=1):
        """
        Returns all configurations where the same duty-cycle length is used for all contexts.

        :param max_cycle: the maximum desired duty-cycle length for the configurations to be generated
        :param name: None or relative path to a file. In latter case the path will be used to save
                   the generated configurations and trade-offs
        :param setting: the setting that is in use while duty-cycling. If None is provided the
                        function assumes that the classification accuracy is 100% and the energy
                        cost is 1
        :param active: the length of the active period. For most purposes this length should be left
                       as the default of 1
        :return: two lists, the first contains pareto-optimal configurations, the second their
                 evaluations in the form (quality, energy).
        """
        configurations = [(i,)*len(self.contexts) for i in range(1, max_cycle)]
        tradeoffs = self.dca_model(configurations, setting=setting, active=active, max_cycle=max_cycle)
        self.save_solution(configurations, tradeoffs, name)
        return configurations, tradeoffs

    def find_dca_tradeoffs(self, max_cycle=10, active=1, seeded=True, setting=None, cf=None, name=None,
                           energy_costs=None, ngen=200):
        """
        Attempts to find the best DCA trade-offs for the current dataset.

        :param max_cycle: the maximum desired duty-cycle length for the configurations
        :param active: the length of the active period. For most purposes this length should be left
                       as the default of 1
        :param seeded: if true, the search starts always start with two specific configuration in
                       the starting population. One configuration uses the mininum and the other the
                       maximum duty-cycle length
        :param setting: the setting that is in use while duty-cycling. If None is provided the
                        function assumes that the classification accuracy is 100% and the energy
                        cost is 1
        :param cf: uses the given confusion matrix (must be normalized, so each row sums to 1)
                   instead of the one prescribed by the ``setting`` parameter
        :param name: None or relative path to a file. In latter case the path will be used to save
                   the generated configurations and trade-offs
        :param energy_costs: internal parameter (its value should not be changed)
        :param ngen: the number of generations in the NSGA_II search
        :return: two lists, the first contains pareto-optimal configurations, the second their
                 evaluations in the form (quality, energy).
        """
        cf = self._calc_duty_cf(setting, cf)
        prob, sleeping, working = dca.precalculate(self.transitions, max_cycle + 1, active)
        mx = self.sensors_on
        if (setting is not None) and (self.setting_to_energy is not None):
            mx = self.setting_to_energy[setting]
        if energy_costs is None:
            energy_costs = mx

        def evl(x):
            return dca.dca_evaluation(transitions=self.transitions, lengths=x, active=active, confusion=cf,
                                      evaluator=self.quality_metric, prob=prob, sleeping_exp=sleeping,
                                      working_exp=working, energy_off=self.sensors_off, energy_costs=energy_costs)

        configurations = dca.dca_find_tradeoffs(len(self.contexts), max_cycle, evl, seeded=seeded, NGEN=ngen)
        tradeoffs = [evl(h) for h in configurations]
        configurations = [list(h) for h in configurations]
        self.save_solution(configurations, tradeoffs, name)
        return configurations, tradeoffs

    def find_dca_random(self, n_samples=100, max_cycle=10):
        """
        Returns ``n_samples`` random DCA configurations.

        :param n_samples: number of configurations to return
        :param max_cycle: the maximum desired duty-cycle length for the configurations
        :return: a list of configurations
        """
        configs = []
        for i in range(n_samples):
            configs.append([choice(range(1, max_cycle)) for _ in range(len(self.contexts))])
        return configs

    def _calc_duty_cf(self, setting, cf):
        if (cf is None) and (setting is not None) and (self.setting_to_confusion is not None):
            cf = self.setting_to_confusion[setting]
        if cf is not None:
            cf = util.normalize_matrix(cf, rows=True)
        return cf

    def sca_dca_real(self, configurations, active=1, name=None, cstree_energy=False):
        """
        Tests the SCA-DCA configurations using a simulation.

        This combines the simulations from (see :func:`~EnergyOptimizer.sca_real`) and
        (see :func:`~EnergyOptimizer.sca_real`). It is slower than using the mathematical model
        (see :func:`~EnergyOptimizer.sca_dca_model`).

        :param configurations: a list of SCA-DCA configurations. A configuration can have the same
                               syntax as the SCA configuration (see :func:`~EnergyOptimizer.sca_real`)
                               or it can be represented as a tuple, where the first element is a
                               SCA configuration and the second element is a DCA configuration
                               (see :func:`~EnergyOptimizer.sca_real`)
        :param active:  the length of the active period. For most purposes this length should be left
                        as the default of 1
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :param cstree_energy: this flag slightly increases the accuracy of the simulation when using
                              the cost-sensitive decision trees. In order to use it, the energy
                              sequence  must be precalculated, by using ``cstree_energy`` flag when
                              generating the trees (e.g. in
                              :func:`~EnergyOptimizer.add_csdt_weighted` or
                              :func:`~EnergyOptimizer.add_csdt_borders`)
        :return: a list of trade-offs, in the form (quality, energy). i-th element represents the
                 evaluation of the i-th configuration
        """
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        sols = [self._sca_dca_real_single(config, active, sequence_true, cstree_energy) for config in configurations]
        self.save_solution(configurations, sols, name)
        return sols

    def _sca_dca_real_single(self, config, active, sequence_true, cstree_energy=None):
        if type(config) == tuple:
            energy_costs = [self.setting_to_energy[s] for s in config[0]]
            sequences = [self.setting_to_sequence[s].apply(lambda x: self.contexts.index(x)) for s in config[0]]
            energy_sequences = None
            if cstree_energy:
                energy_sequences = [self.setting_to_energy_sequence[s] for s in config[0]]
            return test.test_sca_dca(sequence_true, sequences, config[1], energy_costs,
                                       self.sensors_off, self.quality_metric, active, energy_sequences)
        else:
            return self.sca_real(config)[0]

    def sca_dca_model(self, configurations, active=1, cstree=False, name=None):
        """
        Tests the SCA-DCA configurations using a simulation.

        This combines the simulations from (see :func:`~EnergyOptimizer.sca_real`) and
        (see :func:`~EnergyOptimizer.sca_real`). It is slower than using the mathematical model
        (see :func:`~EnergyOptimizer.sca_dca_model`).

        :param configurations: a list of SCA-DCA configurations. A configuration can have the same
                               syntax as the SCA configuration (see :func:`~EnergyOptimizer.sca_real`)
                               or it can be represented as a tuple, where the first element is a
                               SCA configuration and the second element is a DCA configuration
                               (see :func:`~EnergyOptimizer.sca_real`)
        :param active:  the length of the active period. For most purposes this length should be left
                        as the default of 1
        :param cstree: a flag indicating that cost-sensitive trees are used. Makes the evaluation
                       more accurate if used
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :return: a list of trade-offs, in the form (quality, energy). i-th element represents the
                 evaluation of the i-th configuration
        """
        sols = [self._sca_dca_model_single(h, active=active, cstree=cstree) for h in configurations]
        self.save_solution(configurations, sols, name)
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

    def find_sca_dca_tradeoffs(self, sca_configurations=None, sca_tradeoffs=None, dca_indices=None, n_points=5,
                               binary_representation=False, name=None, max_cycle=10, active=1,
                               verbose=False, cstree=False):
        """
        Attempts to find the best SCA-DCA trade-offs for the current dataset.

        :param sca_tradeoffs: if Pareto-optimal trade-ofs were already precalculated using
                              :func:`~EnergyOptimizer.find_sca_tradeoffs` they can be set using this
                              parameter to avoid calculating them again
        :param sca_configurations: if ``sca_tradeoffs`` parameter is set, this parameter should
                                   list the configurations from which the ``sca_tradeoffs`` were
                                   generated
        :param dca_indices: index of the configurations in ``sca_configurations`` to be expanded
                            using the DCA method. If not set, indices will be determined
                            automatically by making them equidistant
        :param n_points: the number of sca configurations selected for expanding with the dca method
        :param binary_representation: a flag indicating that settings are represented by a binary
                                      list. In this case, a different mutation/crossover will be used
                                      for the NSGA-II algorithm
        :param name: None or relative path to a file. In latter case the path will be used to save
                           the generated configurations and trade-offs
        :param max_cycle: the maximum desired duty-cycle length for the configurations
        :param active: the length of the active period. For most purposes this length should be left
                       as the default of 1
        :param verbose: if true, the function prints out the current progress
        :param cstree: a flag indicating that cost-sensitive trees are used. Makes the evaluation
                       more accurate if used
        :return: two lists, the first contains pareto-optimal configurations, the second their
                 evaluations in the form (quality, energy).
        """
        if sca_tradeoffs is None:
            if verbose:
                print("Calculating SCA trade-offs...")
            sca_configurations, sca_tradeoffs = self.find_sca_tradeoffs(binary_representation)
        if dca_indices is None:
            if verbose:
                print("Searching for suitable DCA starting points...")
            start_tradeoffs, start_points = self._calc_sca_dca_points(sca_configurations, sca_tradeoffs, n_points)
            if verbose:
                print("Points found: " + str(len(start_points)))
        else:
            start_tradeoffs = [sca_configurations[i] for i in dca_indices]
            start_points = [sca_tradeoffs[i] for i in dca_indices]
        all_points = []
        pareto_points = [(1 - a, e, h) for h, (a, e) in zip(sca_configurations, sca_tradeoffs)]
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

    def find_coh_tradeoffs(self, alphas=None, name=None):
        """
        Finds the SCA configurations based on context transition probabilities.

        While the method behind this is completely different than
        :func:`~EnergyOptimizer.find_sca_tradeoffs`, it returns the same kind of output, including
        the same type of configurations.

        This method is based on paper:
        Gordon, Dawud, Jürgen Czerny, and Michael Beigl.
        "Activity recognition for creatures of habit."
        Personal and ubiquitous computing 18.1 (2014): 205-221.

        :param alphas: a list of different values of parameter alpha to test (0 <= alpha <=1), see
                       the paper for details
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :return:  two lists, the first contains pareto-optimal configurations, the second their
                  evaluations in the form (quality, energy).
        """
        if alphas is None:
            alphas = np.arange(0, 1, 0.01)
        max_setting = sorted(self.quality().items(), key=lambda x: -x[1])[0][0]
        cfgs = alt.creature_of_habit(self.contexts, self.transitions, self.sequence, self.setting_to_sequence,
                                     self.setting_to_energy, self.quality_metric, alphas, max_setting)
        values = self.sca_real(cfgs)
        cfgs, values = util.pareto_solutions(cfgs, values)
        self.save_solution(cfgs, values, name)
        return cfgs, values

    def find_simple_tradeoffs(self, name=None):
        """
        Finds the SCA configurations based on simple mathematical model.

        It works fast, but the results are often much worse than the alternatives.
        Finds either one or two trade-offs.

        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :return:  two lists, the first contains pareto-optimal configurations, the second their
                  evaluations in the form (quality, energy).
        """
        cfgs = alt.simple_solutions(self.contexts, self.sequence, self.setting_to_sequence, self.setting_to_energy,
                                    self.quality_metric)
        values = self.sca_real(cfgs)
        cfgs, values = util.pareto_solutions(cfgs, values)
        self.save_solution(cfgs, values, name)
        return cfgs, values

    def find_aimd_tradeoffs(self, increase_range=None, decrease_range=None, name=None, active=1):
        """
        Solves the duty-cycle assignment problem in a different ways than the DCA methods

        Imagine a system with parameters ``inc`` and ``dec`` that works as follows:
        1.) System duty-cycles with a cycle of length ``len``.
        2.) If the next detected context is the same as the previous one: ``len = len + inc``.
        3.) If the next detected context is different than the previous one: ``len = len * dec``.
        This method finds good combinations of ``inc`` and ``dec`` from a specified range.

        This method is based on paper:
        Au, Lawrence K., et al.
        "Episodic sampling: Towards energy-efficient patient monitoring with wearable sensors."
        2009 Annual International Conference of the IEEE Engineering in Medicine and Biology Society.
        IEEE, 2009.

        :param increase_range: a list of possible ``inc`` values
        :param decrease_range: a list of possible ``dec`` values
        :param name: None or relative path to a file. In latter case the path will be used to save
                     the generated configurations and trade-offs
        :param active: the length of the active part of the duty-cycle
        :return: two lists, the first contains pareto-optimal configurations in the form
                 (``inc``, ``dec``), the second their evaluations in the form (quality, energy).
        """
        if increase_range is None:
            increase_range = np.arange(0, 15, 0.5)
        if decrease_range is None:
            decrease_range = np.arange(0, 1, 0.05)
        max_setting = sorted(self.quality().items(), key=lambda x: -x[1])[0][0]
        cfgs, values = alt.episodic_sampling(self.sequence, self.setting_to_sequence[max_setting],
                                             energy_off=self.sensors_off, energy_on=self.setting_to_energy[max_setting],
                                             active=active, dec=decrease_range, inc=increase_range,
                                             performance=self.quality_metric)
        cfgs, values = util.pareto_solutions(cfgs, values)
        self.save_solution(cfgs, values, name)
        return cfgs, values

    def find_datumwise_tradeoffs(self):
        return None

    def add_subsets(self, x1, y1, x2, y2, classifier, setting_to_energy=None, setting_fn_energy=None, subsettings=None,
                    subsetting_to_features=None, n=0, feature_groups=None, setting_fn_features=None, name=None,
                    y_p=None, x_p=None, csdt=False, csdt_fn_energy=None, cstree_energy=False):
        """
        Automatically generates settings by taking different attribute subsets from a database

        This method assumes you have a pandas dataframe where each column represent a different
        attribute. Each row represents an instance to be classify. However, each attribute has a cost
        and thus the goal is to classify the instances with as small subset as possible. The cost
        is often shared between the attributes: e.g. if two attributes are calculated from the GPS
        stream, then having one or both has the same cost (the cost of having the GPS open). It may
        also be the case that using some sensor increases or decreases the cost of another sensor
        as they share resources when used.

        There are several ways of specifying which subsets to transform into settings.
        Use the one most convenient for the current domain. In all examples
        we will assume that data comes from different sensors (subsettings) and a setting is
        set of sensors that is in use. If the data (and costs) comes from different sources a
        similar logic applies.

        1. Set ``subsettings`` parameter as a list of sensors and ``subsetting_to_features`` as a
           dictionary that maps each sensor to attribute list (attributes calculated from that
           sensor)
        2. Set ``feature_groups`` as a list of lists. Each list represents features that should all be
           included or excluded in any attribute subset used (e.g. they come from the same sensor).
           Set ``n`` as the number of ``feature_groups``.
        3. Set ``n`` as the number of sensors, set ``setting_fn_features`` as a function that takes
           a binary string of length ``n`` and outputs a list of attributes. Character i in
           binary string represents whether i-th sensor is active.

        This method is not a substitute for attribute selection.

        :param x1: a pandas dataframe of attributes to be used as the training set
        :param y1: a pandas series of labels for the instances in ``x1``
        :param x2: a pandas dataframe of attributes to be used as the testing set
        :param y2: a pandas series of labels for the instances in ``x2``
        :param x_p: (optional) a pandas dataframe of attributes to be used as the pruning set
        :param y_p: (optional) a pandas series of labels for the instances in ``x_p``
        :param classifier: any classifier (tested with sklearn and CS-DTs). Should provide functions
                           "fit" and "predict"
        :param setting_to_energy:  a dictionary, where the keys are the possible settings and the
                                   values are the energy costs (per time unit) when using that setting.
                                   Settings are binary string
        :param setting_fn_energy: a function to be used if the ``setting_to_energy`` is None.
                                  This function should take a setting as the input and return energy
                                  costs (per time unit) when using that setting.
                                  Settings are binary string
        :param subsettings: the list of subsettings
        :param subsetting_to_features: a dictionary that maps subsettings to attribute sets
        :param n: the number of subsettings
        :param feature_groups: a list of lists, elements of the sublists are always all included or
                               excluded
        :param setting_fn_features: a function that maps settings to attribute sets
        :param name: None or relative path to a file. If latter, the generated settings will be saved
        :param x_p: (optional) a pandas dataframe of attributes to be used as the pruning set
        :param y_p: (optional) a pandas series of labels for the instances in ``x_p``
        :param csdt: if the classifier is a CS-DT set this flag to true
        :param csdt_fn_energy: a function used for measuring energy while testing a CS-DT, if it need to be
                               different than the one used for training
        :param cstree_energy: If using CS-DTs it generates accurate energy sequences that can be used
                              to increase the
                              accuracy of the :func:`~EnergyOptimizer.sca_real`. It makes this
                              function much slower
        :return: a list of generated classifiers
        """
        config = gen.general_subsets(x1, y1, x2, y2, classifier, self.contexts, self.proportions, setting_to_energy,
                                     setting_fn_energy, subsettings, subsetting_to_features, n, feature_groups,
                                     setting_fn_features, y_p, x_p, csdt, csdt_fn_energy, cstree_energy)
        self._initialize_settings()
        setting_to_sequence, setting_to_energy, setting_to_energy_matrix, \
            setting_to_energy_sequence, classifiers = config
        self._update_settings(setting_to_sequence, setting_to_energy, setting_to_energy_matrix,
                              setting_to_energy_sequence, name)
        return classifiers

    def add_csdt_weighted(self, cs_tree, x1, y1, x2, y2, x_p=None, y_p=None, test_fn=None, verbose=True,
                          weights_range=None, energy_range=None, n_tree=15, name=None, cstree_energy=False):
        """
        Generates different CS-DTs with different ratios between energy and classification quality.

        When generating cost-sensitive decision trees, at each node a decision is made on whether an
        attribute is worth using (based on its informativeness and cost). Different weights can be
        used to skew the decision in one way or another. Generated CS-DT get added to the list
        of possible settings.

        :param cs_tree: the base cost-sensitive tree object
                        (:class:`eecr.cstree.CostSensitiveTree`) to use for generation of
                        its variants
        :param x1: a pandas dataframe of attributes to be used as the training set
        :param y1: a pandas series of labels for the instances in ``x1``
        :param x2: a pandas dataframe of attributes to be used as the testing set
        :param y2: a pandas series of labels for the instances in ``x2``
        :param x_p: (optional) a pandas dataframe of attributes to be used as the pruning set
        :param y_p: (optional) a pandas series of labels for the instances in ``x_p``
        :param test_fn: a function used for measuring energy while testing a CS-DT, if it need to be
                        different than the one used for training
        :param verbose: if true, the function prints out the current progress
        :param weights_range: a tuple with two elements, representing the minimum and maximum weight
                              to be used. Trees
                              will use ``n_tree`` different weights equidistantly sampled between
                              these two extreme values. Sensible weight range must be determined
                              manually, but as a heuristic, it is usually inversely proportional to
                              the energy costs. For example if energy costs are all >1 then weights
                              should all be <1.
        :param energy_range: An alternative to the ``weights_range`` parameter, setting this to x
                             is equivalent to setting ``weights_range`` to 1/n
        :param n_tree: number of different CS-DT to be generated
        :param name: None or relative path to a file. If latter, the generated settings will be saved
        :param cstree_energy: Generate accurate energy sequences that can be used to increase the
                              accuracy of the :func:`~EnergyOptimizer.sca_real`. It makes this
                              function much slower
        :return: the set of generated CS-DTs
        """
        config = gen.cstree_weighted(cs_tree, x1, y1, x2, y2, self.contexts, x_p, y_p, test_fn, weights_range,
                                     energy_range, n_tree, cstree_energy, verbose)
        self._initialize_settings()
        setting_to_sequence, setting_to_energy, setting_to_energy_matrix, setting_to_energy_sequence, trees = config
        self._update_settings(setting_to_sequence, setting_to_energy, setting_to_energy_matrix,
                              setting_to_energy_sequence, name)
        return trees

    def add_csdt_borders(self, cs_tree, x1, y1, x2, y2, x_p=None, y_p=None, test_fn=None,
                         buffer_range=1, name=None, cstree_energy=False, verbose=False,
                         weights_range=None, energy_range=None, n_tree=15):
        """
        Generates different CS-DTs from data around different contexts.

        When generating these cost-sensitive decision trees, only data from one contexts
        (and instances around it) are taken.  This should create a tree that is good at recognizing
        that context and contexts to which it frequently transitions to. This method also tries
        different weights, same as the :func:`~EnergyOptimizer.add_csdt_weighted` method.
        Generated CS-DT get added to the list of possible settings.

        :param cs_tree: the base cost-sensitive tree object
                        (:class:`eecr.cstree.CostSensitiveTree`) to use for generation of
                        its variants
        :param x1: a pandas dataframe of attributes to be used as the training set
        :param y1: a pandas series of labels for the instances in ``x1``
        :param x2: a pandas dataframe of attributes to be used as the testing set
        :param y2: a pandas series of labels for the instances in ``x2``
        :param x_p: (optional) a pandas dataframe of attributes to be used as the pruning set
        :param y_p: (optional) a pandas series of labels for the instances in ``x_p``
        :param buffer_range: specifies how many instances to take after a contexts transitions
        :param test_fn: a function used for measuring energy while testing a CS-DT, if it need to be
                        different than the one used for training
        :param verbose: if true, the function prints out the current progress
        :param weights_range: a tuple with two elements, representing the minimum and maximum weight
                              to be used. Trees
                              will use ``n_tree`` different weights equidistantly sampled between
                              these two extreme values. Sensible weight range must be determined
                              manually, but as a heuristic, it is usually inversely proportional to
                              the energy costs. For example if energy costs are all >1 then weights
                              should all be <1.
        :param energy_range: An alternative to the ``weights_range`` parameter, setting this to x
                             is equivalent to setting ``weights_range`` to 1/n
        :param n_tree: number of different CS-DT to be generated
        :param name: None or relative path to a file. If latter, the generated settings will be saved
        :param cstree_energy: Generate accurate energy sequences that can be used to increase the
                              accuracy of the :func:`~EnergyOptimizer.sca_real`. It makes this
                              function much slower
        :return: the set of generated CS-DTs
        """
        config = gen.cstree_borders(cs_tree, x1, y1, x2, y2, self.contexts, x_p, y_p, test_fn,
                                    buffer_range, cstree_energy,
                                    weights_range, energy_range, n_tree, verbose)
        self._initialize_settings()
        setting_to_sequence, setting_to_energy, setting_to_energy_matrix, setting_to_energy_sequence, trees = config
        self._update_settings(setting_to_sequence, setting_to_energy, setting_to_energy_matrix,
                              setting_to_energy_sequence, name)
        return trees
