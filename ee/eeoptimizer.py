from ee import dcamodel as dca, eeutility as util, musearch as mu, scamodel as sca

import ee.scamodel2 as s2

from random import choice

import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np

class EnergyOptimizer:
        
    def __init__(self, sequence = None, contexts = None, setting_to_energy=None, setting_to_sequence=None, quality_metric=None,
                path = None):
        self.proportions = None
        self.sequence = None
        self.setting_to_confusion = None
        self.transitions = None
        self.avg_lengths = None
        self.sensors_off = 0
        self.sensors_on = 1
        self.sca_problem = None
        self.sca_solver = None
        self.setting_to_energy = None
        self.setting_to_sequence = None
        self.sca_enum_problem = None
        self.sca_enum_solver = None
        self.sca_bin_problem = None
        self.sca_enum_cf_problem = None
        self.sca_bin_solver = None
        self.sca_enum_eng_problem = None
        self.sca_enum_eng_solver = None
        self.sca_bin_eng_problem = None
        self.sca_bin_eng_solver = None
        self.quality_metric = sca.accuracy_matrix
        self.setting_to_sequence = None
        self.path = "./"
        self.contexts = contexts
        self.setting_to_energy_matrix = None
        self.setting_to_energy_sequence = None
        self.settings = None
        if not path is None:
            self.path = path
        if not quality_metric is None:
            self.quality_metric = quality_metric
        if not sequence is None:
            self.sequence = pd.Series(sequence).reset_index(drop=True)
            self.calc_contexts()
            self.calc_transitions()
        if setting_to_sequence is not None:
            self.set_settings(setting_to_sequence, setting_to_energy)
    
    def set_sequence(self, sequence):
        self.sequence = pd.Series(sequence).reset_index(drop=True)
        self.calc_contexts()
        self.calc_transitions()
        
    def calc_contexts(self):
        if self.contexts is None:
            self.contexts = sorted(self.sequence.unique())
    
    def load_data_config(self, data_name, config_name):
        self.load_data(data_name)
        self.load_config(config_name)

    def load_data_csv(self, name):
        sequence = pd.read_csv(self.path+"/"+name, usecols=[1], names=["Index", "Context"])["Context"]
        self.set_sequence(sequence)

    def load_data(self, name):
        dbfile = open(self.path+"/"+name, 'rb')      
        sequence = pickle.load(dbfile)
        dbfile.close() 
        self.set_sequence(sequence)
        
    def save_data(self, name):
        dbfile = open(self.path+"/"+name, 'wb') 
        pickle.dump(self.sequence, dbfile)                      
        dbfile.close() 
    
    def get_sca_format(self):
        if type(self.settings[0])==tuple:
            min_e, max_e = 0,1
        else:
            min_e, max_e = min(self.settings), max(self.settings)
        return "List of "+str(len(self.contexts))+" x " + str(self.settings[0])+", "+str(min_e)+"-"+str(max_e)
    
    def calc_transitions(self):
        self.transitions = util.get_transitions(self.sequence, self.contexts)
        self.proportions = util.get_steady_state(self.transitions)
        self.avg_lengths = util.average_length(self.transitions)
        
    def calc_confusions(self):
        self.setting_to_confusion = {setting :  util.normalizeMatrix(confusion_matrix(self.sequence,pred,self.contexts), rows=True)
                                         for setting, pred in self.setting_to_sequence.items()}
    
    def get_quality(self):
        return {s:self.quality_metric((self.setting_to_confusion[s].T * self.proportions).T) for s in self.settings}  
    
    def calc_enum_problem(self):
        self.sca_enum_problem = sca.Problem(self.contexts, sca.unpack(lambda x: self.setting_to_confusion[self.settings[x]]),
                              self.quality_metric, sca.unpack(lambda x: self.setting_to_energy[self.settings[x]]),
                              self.proportions, self.transitions)
        self.sca_enum_eng_problem = sca.Problem(self.contexts, sca.unpack(lambda x: self.setting_to_confusion[self.settings[x]]),
                              self.quality_metric, sca.unpack(lambda x: self.setting_to_energy_matrix[self.settings[x]]),
                              self.proportions, self.transitions)
        self.sca_enum_eng_problem.energy_by_matrix = True
        self.sca_enum_eng_cf_problem = sca.Problem(self.contexts, sca.unpack(lambda x: self.setting_to_confusion[self.settings[x]]),
                              lambda x:x, sca.unpack(lambda x: self.setting_to_energy_matrix[self.settings[x]]),
                              self.proportions, self.transitions)
        self.sca_enum_eng_cf_problem.energy_by_matrix = True
        self.sca_enum_cf_problem = sca.Problem(self.contexts, sca.unpack(lambda x: self.setting_to_confusion[self.settings[x]]),
                              lambda x:x, sca.unpack(lambda x: self.setting_to_energy[self.settings[x]]),
                              self.proportions, self.transitions)
        evl = sca.unpack(lambda setting, x1, y1, x2, y2: self.setting_to_sequence[self.settings[setting]])
        self.sca_enum_problem.setup_real(create_test=evl, x1 = self.sequence, y1= self.sequence, 
                                         x2 = self.sequence, y2 = self.sequence)
        self.sca_enum_eng_problem.setup_real(create_test=evl, x1 = self.sequence, y1= self.sequence, 
                                         x2 = self.sequence, y2 = self.sequence)
        self.sca_enum_solver = mu.Solver(self.sca_enum_problem, len(self.settings)-1, "index")
        self.sca_enum_eng_solver = mu.Solver(self.sca_enum_eng_problem, len(self.settings)-1, "index")
        self.sca_bin_problem = sca.Problem(self.contexts, sca.unpack(lambda x: self.setting_to_confusion[x]),
                              self.quality_metric, sca.unpack(lambda x: self.setting_to_energy[x]),
                              self.proportions, self.transitions)
        self.sca_bin_eng_problem = sca.Problem(self.contexts, sca.unpack(lambda x: self.setting_to_confusion[x]),
                              self.quality_metric, sca.unpack(lambda x: self.setting_to_energy_matrix[x]),
                              self.proportions, self.transitions)
        evl = sca.unpack(lambda setting, x1, y1, x2, y2: self.setting_to_sequence[setting])
        self.sca_bin_problem.setup_real(create_test=evl, x1 = self.sequence, y1= self.sequence, 
                                         x2 = self.sequence, y2 = self.sequence)
        self.sca_bin_eng_problem.setup_real(create_test=evl, x1 = self.sequence, y1= self.sequence, 
                                         x2 = self.sequence, y2 = self.sequence)
        if type(self.settings[0]) in [tuple, list]:
            self.sca_bin_solver = mu.Solver(self.sca_bin_problem, len(self.settings[0]), "list")
            self.sca_bin_eng_solver = mu.Solver(self.sca_bin_eng_problem, len(self.settings[0]), "list")
        
    
    def set_path(self, path):
        self.path = path
    
    def set_settings(self, setting_to_sequence, setting_to_energy=None, setting_fn_energy=None):
        if setting_to_energy is None and setting_fn_energy is None:
            raise AssertionError("Either setting_to_energy or setting_fn_energy must not be None")
        self.setting_to_energy = setting_to_energy
        if setting_to_sequence is not None:
            self.setting_to_sequence = {k:pd.Series(v) for (k,v) in setting_to_sequence.items()}
            self.settings = setting_to_sequence.keys()
            if setting_fn_energy is not None:
                self.setting_to_energy = self.calc_setting_transcription(setting_fn_energy)
            self.calc_confusions()
            self.calc_enum_problem()
    
    def set_transitions(self, transitions):
        self.transitions = util.normalizeMatrix(transitions, rows=True)
        self.proportions = util.get_steady_state(self.transitions)
        self.avg_lengths = util.average_length(self.transitions)
        self.calc_enum_problem()
        

    def calc_setting_transcription(self, setting_fn_energy):
        return {setting : setting_fn_energy(setting) for setting in self.settings}
    
    def get_summary(self):
        #print "Proportions"
        summary = pd.DataFrame()
        summary["Proportions"] = pd.Series(self.proportions,self.contexts)
        summary["Average lengths"] = pd.Series(self.avg_lengths,self.contexts)
        print (summary)
        #print "Transitions"
        #print self.transitions
    
    def set_dca_costs(self, cost_off, cost_on):
        self.sensors_on = cost_on
        self.sensors_off = cost_off
    
    def get_dca_static(self, max_cycle, name = None, setting = None, active = 1):
        hofs = [(i,)*len(self.contexts) for i in range(1,max_cycle)]
        sols = [self.get_dca_evaluation(hof, setting = setting, active = active) for hof in hofs]
        if name is not None: 
            self.save_solution(hofs, sols, name)
        return (hofs, sols)
    
    def get_dca_evaluation(self, x, active = 1, setting = None, cf = None, energy_costs = None):
        if len(x)!=len(self.contexts):
            raise ValueError("Assignment length does not match the number of contexts!") 
        cf = self.calc_duty_cf(setting, cf)
        if energy_costs is None:
            evl = lambda x: evalDutyCycle(x, transition = self.transitions, active = active, cf = cf, 
                                          performance = self.quality_metric)
            sol = evl(x)
            return sol[0], self.calc_duty_cost(setting, sol[1]), sol[1]
        else: 
            prob, sleeping, working = dca.precalculate(self.transitions,max(x)+1,cf,active)
            evl = lambda x: evalDutyCycleFastCosts(x, transition = self.transitions, active = active, cf = cf, 
                                    performance = self.quality_metric, prob=prob, sleeping=sleeping, working=working,
                                    energy_costs = energy_costs, min_cost=self.sensors_off)
            sol = evl(x)
            return sol[0], -sol[1], -sol[1]
        
    ##TODO revise return, save, itd
    def get_model_dca(self, hof, active = 1, setting = None, cf = None, max_cycle = None):
        cf = self.calc_duty_cf(setting, cf)
        if max_cycle is None:
            max_cycle = max([max(config) for config in hof])
        prob, sleeping, working = dca.precalculate(self.transitions,max_cycle+1,cf,active)
        evl = lambda x: evalDutyCycleFast(x, transition = self.transitions, active = active, cf = cf, 
                            performance = self.quality_metric, prob=prob, sleeping=sleeping, working=working )
        sols = []
        for config in hof:
            a, e = evl(config)
            sols.append((a, self.calc_duty_cost(setting,e), e))
        return sols
    
    def get_dca_tradeoffs(self, max_cycle = 10, active = 1, seeded = True, setting = None, cf = None, name = None, 
                         fast = True, energy_costs = None, NGEN=200):
        cf = self.calc_duty_cf(setting, cf)
        #evl = lambda x: evalDutyCycle(x, transition = self.transitions, active = active, cf = cf, 
        #                              performance = self.quality_metric)
        prob, sleeping, working = dca.precalculate(self.transitions,max_cycle+1,cf,active)
            #print  prob, sleeping, working
        evl = lambda x: evalDutyCycleFast(x, transition = self.transitions, active = active, cf = cf, 
                                    performance = self.quality_metric, prob=prob, sleeping=sleeping, working=working )
        if energy_costs is not None:
            evl = lambda x: evalDutyCycleFastCosts(x, transition = self.transitions, active = active, cf = cf, 
                                    performance = self.quality_metric, prob=prob, sleeping=sleeping, working=working,
                                    energy_costs = energy_costs, min_cost=self.sensors_off)
            
        hof = dca.geneticSolutions(len(self.contexts), max_cycle, evl, seeded=seeded, NGEN=NGEN)
        solutions = [evl(h) for h in hof]
        if energy_costs is None:
            solutions = [(a, self.calc_duty_cost(setting,e), e)  for (a,e) in solutions]
        else:
            solutions = [(a, -e, -e)  for (a,e) in solutions]
        hof = [list(h) for h in hof] 
        if name is not None: 
            self.save_solution(hof, solutions, name)
        return hof, solutions
    
    def calc_duty_cf(self, setting, cf):
        if (cf is None) and (setting is not None) and (self.setting_to_confusion is not None):
            cf = self.setting_to_confusion[setting]
        if cf is not None:
            cf = util.normalizeMatrix(cf, rows=True)
        return cf
    
    def calc_duty_cost(self,setting, gain):
        mx = self.sensors_on
        if (setting is not None) and (self.setting_to_energy is not None):
            mx = self.setting_to_energy[setting]
        return dca.duty_energy(gain+1, self.sensors_off, mx)
    
    def create_problem(self):
        problem = sca.Problem(self.contexts, lambda x: self.setting_to_confusion[x], self.quality_metric,
                          lambda x: self.setting_to_energy[x], self.proportions, self.transitions)
        return problem
    
    def encrypt_hof(self, hof):
        return [[self.settings[sett] for sett in assignment] for assignment in hof]
    
    def decrypt_hof(self, hof):
        return [[self.settings.index(sett) for sett in assignment] for assignment in hof]

    def encrypt_solution(self, solution):
        return [self.settings[sett] for sett in solution]

    def decrypt_solution(self, solution):
        return [self.settings.index(sett) for sett in solution]

    def sca_simple(self, tradeoffs):
        tradeoffs = self._wrap(tradeoffs)
        return [self._sca_simple_single(configuration) for configuration in tradeoffs]

    def _sca_simple_single(self, configuration):
        self._checks(configuration)
        confusions = [self.setting_to_confusion[setting] for setting in configuration]
        energies = [self.setting_to_energy[setting] for setting in configuration]
        return s2.sca_simple_evaluation(self.proportions, confusions, energies, self.quality_metric)

    def sca_model(self, tradeoffs, encrypted=False):
        tradeoffs = self._wrap(tradeoffs)
        if encrypted:
            tradeoffs = self.encrypt_hof(tradeoffs)
        return [self._sca_model_single(configuration) for configuration in tradeoffs]

    def _sca_model_single(self, configuration, return_cf=False):
        self._checks(configuration)
        confusions = [self.setting_to_confusion[setting] for setting in configuration]
        energies = [self.setting_to_energy[setting] for setting in configuration]
        quality = (lambda x: x) if return_cf else self.quality_metric
        return s2.sca_evaluation(self.transitions, confusions, energies, quality)

    def find_sca_tradeoffs(self, solution_type="enumerate", name=None, energy_type="default"):
        if self.setting_to_confusion is None or self.setting_to_energy is None:
            raise AssertionError("Confusion matrices or energy estimations missing!")
        if solution_type not in ["enumerate", "binary"]:
            raise ValueError("Solution type can be either 'enumerate' or 'binary'")
        if solution_type == "enumerate":
            tradeoffs = s2.sca_find_tradeoffs(solution_type, len(self.settings), len(self.contexts), self, NGEN=200)
            tradeoffs = self.encrypt_hof(tradeoffs)
        if solution_type == "binary":
            tradeoffs = s2.sca_find_tradeoffs(solution_type, len(self.settings[0]), len(self.contexts), self, NGEN=200)
        values = self.sca_model(tradeoffs)
        if name is not None:
            self.save_solution(tradeoffs, values, name)
        return tradeoffs, values

    @staticmethod
    def _wrap(tradeoffs):
        if type(tradeoffs[0]) != list:
            tradeoffs = [tradeoffs]
        return tradeoffs

    def _checks(self, configuration):
        if self.setting_to_confusion is None or self.setting_to_energy is None:
            raise AssertionError("Confusion matrices or energy estimations missing!")
        if len(configuration) != len(self.contexts):
            raise ValueError("Assignment length does not match the number of contexts!")

    def get_sca_evaluation(self, setting, solution_type = "enumerate", cf = False, energy_type="default"):
        if self.setting_to_confusion is None or self.setting_to_energy is None:
            raise AssertionError("Settings are not set")
        if len(setting)!=len(self.contexts):
            raise ValueError("Assignment length does not match the number of contexts!") 
        if solution_type == "enumerate":
            setting = [self.settings.index(x) for x in setting]
            problem = self.sca_enum_problem
            if cf:
                problem = self.sca_enum_cf_problem
            if energy_type == "csdt":
                problem = self.sca_enum_eng_problem
            if cf and energy_type == "csdt":
                problem = self.sca_enum_eng_cf_problem
            config = sca.listToConfiguration(setting,problem)
            return problem.markovQualityEnergy(config)
        
    def find_sca_static(self, name = None):
        if self.setting_to_sequence is None or self.setting_to_energy is None:
            raise AssertionError("Settings are not set")
        #h = self.settings
        points = []
        for s in self.settings:
            cf = confusion_matrix(self.sequence, self.setting_to_sequence[s], self.contexts)
            e = self.setting_to_energy[s]
            a = self.quality_metric(cf)
            points.append((1-a,e,s))
        pareto_points = util.pareto_dominance(points)
        pareto_hof = list(zip(*pareto_points))[2]
        pareto_sols = mu.reverse_first(zip(*list(zip(*pareto_points))[0:2]))
        if name is not None: 
            self.save_solution(pareto_hof, pareto_sols, name)
        return (pareto_hof, pareto_sols)
    
    def find_sca_random(self, n_samples=100):
        configs = []
        for i in range(n_samples):
            configs.append([choice(self.settings) for _ in range(len(self.contexts))])
        return configs
    
    def find_dca_random(self, n_samples=100, max_cycles = 10):
        configs = []
        for i in range(n_samples):
            configs.append([choice(range(1,max_cycles)) for _ in range(len(self.contexts))])
        return configs
    
    def get_sca_tradeoffs(self, solution_type = "enumerate", name = None, energy_type="default"):
        if self.setting_to_confusion is None or self.setting_to_energy is None:
            raise AssertionError("Settings are not set")
        if solution_type not in ["enumerate", "binary"] :
            raise ValueError("Solution type can be either 'enumerate' or 'binary'")
        if solution_type == "enumerate":
            problem, solver = self.sca_enum_problem, self.sca_enum_solver
            if energy_type == "csdt":
                problem, solver = self.sca_enum_eng_problem, self.sca_enum_eng_solver
            solver.solve(NGEN=200)
            hof = self.encrypt_hof(solver.hof)
            solutions = [solver.evalAssignment(p, problem) for p in solver.hof]
        if solution_type == "binary":
            problem, solver = self.sca_bin_problem, self.sca_bin_solver
            if energy_type == "csdt":
                problem, solver = self.sca_bin_eng_problem, self.sca_bin_eng_solver
            solver.solve(NGEN=200)
            num_subsettings = len(self.settings[0])
            hof = [[tuple(setting[i*num_subsettings : (i+1)*num_subsettings]) 
                    for i in range(len(self.contexts))] for setting in solver.hof]
            solutions = [solver.evalAssignment(p, problem) for p in solver.hof]
        solutions = [(float(a),float(e)) for (a,e) in solutions]
        if name is not None: 
            self.save_solution(hof, solutions, name)
        return hof, solutions
    
    def get_model_sca(self, hof, energy_type = "default"):
        return [self.get_sca_evaluation(p, energy_type=energy_type) for p in hof]
    
    def get_naive_sca(self, hof):
        hof = self.decrypt_hof(hof)
        acc = [self.sca_enum_problem.simpleQuality2(sca.listToConfiguration(p,self.sca_enum_problem)) for p in hof]
        energy = [self.sca_enum_problem.simpleEnergy(sca.listToConfiguration(p,self.sca_enum_problem)) for p in hof]
        return zip(acc,energy)
        
    def get_real_sca(self, hof, name = None):
        hof = self.decrypt_hof(hof)
        solutions = [self.sca_enum_problem.tester_fast(sca.listToConfiguration(p,self.sca_enum_problem), special=True) for p in hof]
        if name is not None:
            self.save_solution(hof, solutions, name)
        return solutions
    
    def get_real_sca_alt(self, hof, name = None, use_energy_sequence=False):
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        solutions = []
        for config in hof:
            sequences = [self.setting_to_sequence[s].apply(lambda x: self.contexts.index(x)) for s in config]
            energy_costs = [self.setting_to_energy[s] for s in config]
            energy_sequences = None
            if use_energy_sequence:
                energy_sequences = [self.setting_to_energy_sequence[s] for s in config]
            solutions.append(dca.test_sca_dca(sequence_true, sequences, [1]*len(self.contexts), energy_costs,
                                self.sensors_off, self.quality_metric, 1, energy_sequences))
        if name is not None:
            self.save_solution(hof, solutions, name)
        return solutions   
    
    def get_real_alt_dca(self, hof, setting = None, cf = None, active = 1, name = None):
        if (cf is None) and (setting is not None) and (self.setting_to_confusion is not None):
            cf = self.setting_to_confusion[setting]
        seq = self.sequence.apply(lambda x: self.contexts.index(x))
        solutions = [dca.duty_test(seq, h, confusion=cf, active=active)[0:2] for h in hof]
        mx = self.sensors_on
        if (setting is not None) and (self.setting_to_energy is not None):
            mx = self.setting_to_energy[setting]
        solutions = [(a, dca.duty_energy(e+1, self.sensors_off, mx), e)  for (a,e) in solutions]
        if name is not None:
            self.save_solution(hof, solutions, name)
        return solutions
    
    def get_real_dca(self, hof, setting = None, active = 1, name = None):
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        if setting is not None:
            sequence_predicted = self.setting_to_sequence[setting].apply(lambda x: self.contexts.index(x))
            energy = self.setting_to_energy[setting]
        else:
            sequence_predicted = sequence_true
            energy = self.sensors_on
        sols = []
        for config in hof:
            sols.append(dca.test_sca_dca(sequence_true, [sequence_predicted]*len(self.contexts), config,
                                     [energy]*len(self.contexts), self.sensors_off, self.quality_metric, active))
        if name is not None:
            self.save_solution(hof, sols, name)
        return sols 
    
    def bin_encode(self, x,length):
        s = str(bin(x))[2:]
        pad = [0]*(length-len(s)) 
        return pad + [int(x) for x in list(s)]
    
    def save_solution(self, hof, solution, name):
        dbfile = open(self.path+"/"+name, 'wb') 
        pickle.dump((hof, solution), dbfile)                      
        dbfile.close() 
    
    def save_config(self, name):
        dbfile = open(self.path+"/"+name, 'wb') 
        pickle.dump((self.settings, self.setting_to_sequence, self.setting_to_energy, 
                     self.setting_to_confusion, self.setting_to_energy_matrix, self.setting_to_energy_sequence), dbfile)                      
        dbfile.close() 
    
    def load_solution(self, name):
        dbfile = open(self.path+"/"+name, 'rb')      
        loaded = pickle.load(dbfile)
        hof, solutions = loaded
        dbfile.close() 
        return hof, solutions

    def expand_tuple(self, t):
        if len(t) <= 3:
            return int(t)
        return tuple([int(x) for x in t])

    def load_config_csv(self, name):
        dbfile = open(self.path+"/"+name+"_rest", 'rb')
        loaded = pickle.load(dbfile, encoding='latin1')
        self.settings, self.setting_to_sequence, self.setting_to_energy, self.setting_to_confusion = loaded[0:4]
        if len(loaded) > 4:
            self.setting_to_energy_matrix = loaded[4]
            self.setting_to_energy_sequence = loaded[5]
        df = pd.read_csv(self.path+"/"+name+"_sequence.csv").drop("Unnamed: 0", axis=1)
        self.setting_to_sequence = {}
        for c in df.columns:
            self.setting_to_sequence[self.expand_tuple(c)] = df[c]
        dbfile.close()
        self.calc_enum_problem()

    def load_config(self, name):
        dbfile = open(self.path+"/"+name, 'rb')      
        loaded = pickle.load(dbfile)
        self.settings, self.setting_to_sequence, self.setting_to_energy, self.setting_to_confusion = loaded[0:4]
        if len(loaded) > 4:
            self.setting_to_energy_matrix = loaded[4]
            self.setting_to_energy_sequence = loaded[5]
        dbfile.close() 
        self.calc_enum_problem()
    
    def set_binary(self, x1, y1, x2, y2, classifier, setting_to_energy=None, setting_fn_energy=None, subsettings = None, 
                   subsetting_to_features=None, n=0, feature_groups = None, setting_fn_features=None, name = None, 
                   verbose = False, y_p=None, x_p = None, csdt = False, csdt_fn_energy = None, use_energy_sequence = False):
        if setting_to_energy is None and setting_fn_energy is None and not csdt: 
            raise AssertionError("Energy parameters missing")
        fill_energy = False
        if setting_to_energy is None:
            fill_energy = True
        parameter_type = 0
        if subsetting_to_features is not None and subsettings is not None:
            parameter_type = 1
        if n!=0 and feature_groups is not None:
            parameter_type = 2
        if n!=0 and setting_fn_features is not None:
            parameter_type = 3
        if parameter_type==0:
            raise AssertionError("Setting parameters missing")
        if n==0:
            n = len(subsettings)
        possibilities = [self.bin_encode(x,n) for x in range(2**n)]
        majority_class = self.contexts[list(self.proportions).index(max(self.proportions))]
        self.setting_to_sequence = {}
        if fill_energy:
            setting_to_energy = {}
            if csdt:
                self.setting_to_energy_matrix = {}
                self.setting_to_energy_sequence = {}
        classifiers = {}
        for setting in possibilities:
            if verbose:
                print(setting)
            if parameter_type==1:
                features = list(set().union(*[subsetting_to_features[key] for (i,key) in enumerate(subsettings) 
                                              if setting[i]==1]))
            if parameter_type==2:
                features = list(set().union(*[feature_groups[i] for i in range(n) if setting[i]==1]))
            if parameter_type==3:
                features = setting_fn_features(setting)
            if len(features)==0 and not csdt:
                predictions = [majority_class]*len(y2)
            else:
                if x_p is None:
                    classifier.fit(x1[features],y1)
                else:
                    classifier.fit(x1[features],y1,x_p[features],y_p)
                predictions = classifier.predict(x2[features])
            self.setting_to_sequence[tuple(setting)] = predictions
            if verbose:
                print(self.quality_metric(confusion_matrix(self.sequence, predictions, self.contexts)))
            if fill_energy:
                if not csdt:
                    setting_to_energy[tuple(setting)] = setting_fn_energy(setting)
                else:
                    energy = classifier.energy(x2, csdt_fn_energy)
                    energy_matrix = [classifier.energy(x2[y2==act], csdt_fn_energy) for act in self.contexts]
                    if use_energy_sequence:
                        energy_sequence = [classifier.energy(x2.iloc[i:i+1]) for i in range(len(x2))]
                        self.setting_to_energy_sequence[tuple(setting)] = energy_sequence
                    self.setting_to_energy_matrix[tuple(setting)] = energy_matrix
                    setting_to_energy[tuple(setting)] = energy
                    classifiers[tuple(setting)] = classifier
        self.set_settings(self.setting_to_sequence, setting_to_energy)
        if name is not None:
            self.save_config(name)
        #if csdt:
        #    return classifiers
    
    def get_real_sca_dca(self, hof, active = 1, name = None, use_energy_sequence = False):
        real = []
        sequence_true = self.sequence.apply(lambda x: self.contexts.index(x))
        for config in hof:
            if type(config)==tuple:
                energy_costs = [self.setting_to_energy[s] for s in config[0]]
                sequences = [self.setting_to_sequence[s].apply(lambda x: self.contexts.index(x)) for s in config[0]]
                energy_sequences = None
                if use_energy_sequence:
                    energy_sequences = [self.setting_to_energy_sequence[s] for s in config[0]]
                real.append(dca.test_sca_dca(sequence_true, sequences, config[1], energy_costs,
                                self.sensors_off, self.quality_metric, active, energy_sequences))
            else:
                real.append(self.get_real_sca([config])[0])
        return real
        
    
    def get_sca_dca_evaluation(self, setting, solution_type = "enumerate", active = 1, energy_type = "default"):
        if type(setting)==tuple:
            cf = self.get_sca_evaluation(setting[0], solution_type=solution_type, cf=True, energy_type=energy_type)[0]
            energy_costs = [self.setting_to_energy[s] for s in setting[0]]
            if energy_type == "csdt":
                energy_costs = [self.setting_to_energy_matrix[row] for row in setting[0]]
            return self.get_dca_evaluation(setting[1], cf = cf, active=active, energy_costs=energy_costs)
        else:
            return self.get_sca_evaluation(setting, solution_type=solution_type)
    
    def get_model_sca_dca(self, hof, active = 1, energy_type = "default"):
        sols = [self.get_sca_dca_evaluation(h, active=active, energy_type=energy_type) for h in hof]
        return sols
    
    def get_sca_dca_tradeoffs(self, hof = None, solutions = None, dca_indices = None, n_points = 5, 
                              solution_type = "enumerate", name = None, max_cycle = 10, active = 1,
                             verbose = False, return_all = False, energy_type = "default"):
        if solutions is None:
            if verbose:
                print ("Calculating SCA trade-offs...")
            hof, solutions = self.get_sca_tradeoffs(solution_type)
        if dca_indices is None:
            if verbose:
                print ("Searching for suitable DCA starting points...")
            start_hofs, start_points = self.calc_sca_dca_points(hof, solutions, n_points)
            #print X
            if verbose:
                print ("Points found: "+str(len(start_points)))
        else:
            start_hof = [hof[i] for i in dca_indices]
            start_points = [solutions[i] for i in dca_indices]
        all_points = []
        pareto_points = [(1-a,e,h) for h, (a,e) in zip(hof, solutions)]
        for (h,point) in zip(start_hofs, start_points):
            if verbose:
                print ("Expanding trade-off: " + str(point))
            energy_costs = [self.setting_to_energy[setting] for setting in h]
            if energy_type == "csdt":
                energy_costs = [self.setting_to_energy_matrix[setting] for setting in h]
            cf = self.get_sca_evaluation(h, cf = True)[0]
            #print cf
            dca_hof, dca_s = self.get_dca_tradeoffs(max_cycle=max_cycle, cf = cf, active=active, energy_costs=energy_costs)
            sca_dca_hof = [(h, dca_h) for dca_h in dca_hof]
            all_points.append((sca_dca_hof, dca_s))
            pareto_points += [(1-a, e, sca_dca_h) for (a,e,e2), sca_dca_h in zip(dca_s, sca_dca_hof)]
        pareto_points = util.pareto_dominance(pareto_points)
        pareto_hof = list(zip(*pareto_points))[2]
        pareto_sols = mu.reverse_first(zip(*list(zip(*pareto_points)[0:2])))
        if name is not None:
            self.save_solution(pareto_hof, pareto_sols, name)
        if return_all:
            return all_points, (pareto_hof, pareto_sols)
        else:
            return pareto_hof, pareto_sols
    
    
    def calc_sca_dca_points(self,hof, solutions, number = 5):
        if number > hof:
            pass
        acc_threshold = (solutions[0][0]-solutions[-1][0])/(2.0*number)
        energy_threshold = (solutions[0][1]-solutions[-1][1])/(2.0*number)
       
        while(True):
            #print acc_threshold, energy_threshold
            points, hofs, x = [], [],  solutions[0][0]+1
            acc_lst, energy_lst = zip(*solutions)
            #print acc_lst
            for i,(acc,energy,h) in enumerate(zip(acc_lst,energy_lst,hof)):
                #return
                if acc < x and (len(points)==0 or energy<points[-1][1]-energy_threshold): 
                    #print acc, energy
                    x-acc_threshold
                    points.append((acc,energy))
                    hofs.append(h)
            if len(points)>number:
                return hofs, points
            else:
                #print len(pointX)
                acc_threshold/=2
                energy_threshold/=2
                
                
    def set_csdt_weighted(self, cs_tree, x1, y1, x2, y2, x_p=None, y_p=None, test_fn = None,
                          weights_range = None, energy_range = None, n_tree = 15, set_settings = False,
                         verbose = False, name = None, use_energy_sequence = False, return_trees = False ):
        if weights_range is None:
            if energy_range is None:
                raise AssertionError("Either weights_range or energy_range must be set")
            else:
                weights_range = (0, 1.0 / (energy_range[1]-energy_range[0]))
        if self.setting_to_sequence is None:
            self.setting_to_sequence = {}
            self.setting_to_energy = {}
            self.setting_to_energy_matrix = {}
            self.setting_to_energy_sequence = {}
        interval = (weights_range[1]-weights_range[0]) / float(n_tree)
        weights = list(np.arange(weights_range[0],weights_range[1],interval))+[weights_range[1]*100]
        trees = {}
        for weight in weights:
            cs_tree.set_weight(weight)
            cs_tree.fit(x1,y1)
            energy = cs_tree.energy(x2, test_fn)
            energy_matrix = [cs_tree.energy(x2[y2==act], test_fn) for act in self.contexts]
            predictions = cs_tree.predict(x2)
            self.setting_to_sequence[weight] = predictions
            self.setting_to_energy[weight] = energy
            self.setting_to_energy_matrix[weight] = energy_matrix
            if return_trees:
                trees[weight] = cs_tree.copy()
            if use_energy_sequence:
                energy_sequence = [cs_tree.energy(x2.iloc[i:i+1]) for i in range(len(x2))]
                self.setting_to_energy_sequence[weight] = energy_sequence
            if verbose:
                quality = self.quality_metric(confusion_matrix(self.sequence, predictions, self.contexts))
                print (weight, quality, energy)
        self.set_settings(self.setting_to_sequence, self.setting_to_energy)
        if name is not None:
            self.save_config(name)
        if return_trees:
            return trees
        #return hof, solutions
        
    def set_csdt_borders(self, cs_tree, x1, y1, x2, y2, x_p=None, y_p=None, test_fn = None,
                         buffer_range = 1, verbose = False, name = None, use_energy_sequence = False,
                        weights_range = None, energy_range = None, n_tree = 15):
        if self.setting_to_sequence is None:
            self.setting_to_sequence = {}
            self.setting_to_energy = {}
            self.setting_to_energy_matrix = {}
            self.setting_to_energy_sequence = {}
        if weights_range is not None or energy_range is not None:
            if weights_range is None:
                weights_range = (0, 1.0 / (energy_range[1]-energy_range[0]))
            interval = (weights_range[1]-weights_range[0]) / float(n_tree)
            weights = list(np.arange(weights_range[0],weights_range[1],interval))+[weights_range[1]*100]
        else:
            weights = [cs_tree.weight]
        for context in self.contexts:
            cs_tree.default = context
            cond = pd.Series([False]*len(y1))
            for i in range(buffer_range+1):
                cond = (cond | (y1.shift(i)==context))
            y1c = y1[cond]
            x1c = x1[cond]
            for weight in weights:
                setting = (context, weight)
                cs_tree.set_weight(weight)
                cs_tree.fit(x1c,y1c)
                energy = cs_tree.energy(x2, test_fn)
                energy_matrix = [cs_tree.energy(x2[y2==act], test_fn) for act in self.contexts]
                predictions = cs_tree.predict(x2)
                self.setting_to_sequence[setting] = predictions
                self.setting_to_energy[setting] = energy
                self.setting_to_energy_matrix[setting] = energy_matrix
                if use_energy_sequence:
                    energy_sequence = [cs_tree.energy(x2.iloc[i:i+1]) for i in range(len(x2))]
                    self.setting_to_energy_sequence[setting] = energy_sequence
                if verbose:
                    quality = self.quality_metric(confusion_matrix(self.sequence, predictions, self.contexts))
                    print (setting, quality, energy)
        self.set_settings(self.setting_to_sequence, self.setting_to_energy)
        if name is not None:
            self.save_config(name)
        


# In[26]:

def evalDutyCycle(x, transition = None, active = None, cf = None, performance = None):
    result = dca.duty_prediction(transition, x, active=active, confusion=cf)
    return (performance(result[2]),result[1])


# In[53]:

def evalDutyCycleFast(x, sleeping = None, working= None, prob = None, 
                      transition = None, active = None, cf = None, performance = None):
    result = dca.duty_predict_fast(transition, x, prob = prob, sleeping_exp = sleeping, working_exp = working,
                          active=active, confusion=cf)
    return (performance(result[2]),result[1])


# In[56]:

def evalDutyCycleFastCosts(x, sleeping = None, working= None, prob = None, 
                      transition = None, active = None, cf = None, performance = None, energy_costs=None, min_cost= 0):
    result = dca.duty_predict_fast(transition, x, prob = prob, sleeping_exp = sleeping, working_exp = working,
                          active=active, confusion=cf, energy_costs = energy_costs)
    return (performance(result[2]), -dca.duty_energy(result[1]+1,min_cost,result[3]))

