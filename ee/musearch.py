
# coding: utf-8

# In[1]:

import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import time

from ee import scamodel as sca

# In[3]:

import pickle


# In[6]:

from deap.tools import History
from collections import defaultdict


class Solver:
    
    use_naive = False
    max_length = 1
    history = None
    
    def evalAssignment(self, lst,problem):
        configuration = sca.listToConfiguration(lst, problem)
        if self.problem.track:
            self.problem.numEvals+=1
            t0 = time.clock()
        q, e = problem.markovQualityEnergy(configuration)
        if self.use_naive:
            q, e = (problem.simpleQuality2(configuration), problem.simpleEnergy(configuration))
        if self.problem.track:
            self.problem.timeEvals+=time.clock()-t0
        return numpy.real(q), numpy.real(e)

#    def listToConfiguration(self,lst,problem):
#        settings = np.reshape(lst,(len(problem.activities),-1))
#        settings = [(s if len(s)>1 else s[0]) for s in settings]
#        #print settings
#        return Configuration(problem, settings)    
        
    def __init__(self,problem, length, setting_type, save_history=False):
        self.problem = problem
        creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.Fitness)
        self.toolbox = base.Toolbox()
        if setting_type == "list":
            self.toolbox.register("attr_bool", random.randint, 0, 1)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                  self.toolbox.attr_bool, length*len(problem.activities))
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        if setting_type == "index":
            self.max_length = length
            self.toolbox.register("attr_bool", random.randint, 0, length)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                    self.toolbox.attr_bool, len(self.problem.activities))
            self.toolbox.register("mutate", mutate_list, indpb=0.05, max_length = length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)    
        self.toolbox.register("evaluate", lambda x : self.evalAssignment(x,problem))
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("select", tools.selNSGA2)
        if save_history:
            self.history = History()
            self.toolbox.decorate("mate", self.history.decorator)
            self.toolbox.decorate("mutate", self.history.decorator)
        
    def load_from_copy(self, solver):
        self.fill()
        self.hof = solver.hof
        self.accN = solver.accN   
        self.energyN = solver.energyN
        self.accM = solver.accM
        self.energyM = solver.energyM  
        self.accR = solver.accR
        self.energyR = solver.energyR
        self.problem = solver.problem
        self.toolbox = solver.toolbox
        
    def solve(self, NGEN = 50, track = False, MU = 50, verbose = False):
        if track: 
            self.problem.track = True
            self.problem.numSettings = 0
            self.problem.numEvals = 0
            self.problem.timeSettings = 0
            self.problem.timeEvals = 0
        LAMBDA = 100
        CXPB = 0.7
        MUTPB = 0.2
        pop = self.toolbox.population(n=MU)
        if self.history:
            self.history.update(pop)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        #stats.register("min", numpy.min, axis=0)
        #stats.register("max", numpy.max, axis=0)
        algorithms.eaMuPlusLambda(pop, self.toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                                  halloffame=hof, verbose=verbose)
        self.hof = hof
        if track:
            if self.problem.numSettings>0:
                self.problem.timeSettings/=self.problem.numSettings/1000.0
            if self.problem.numEvals>0:
                self.problem.timeEvals/=self.problem.numEvals/1000.0

        return pop, stats, hof
    
    def save_hof(self, name):
        dbfile = open(name, 'ab') 
        pickle.dump(self.hof, dbfile)                      
        dbfile.close() 
    
    def load_hof(self, name): 
        dbfile = open(name, 'rb')      
        self.hof = pickle.load(dbfile) 
        dbfile.close() 
    
    def calculateApprox(self, skipSimple=False):
        if not skipSimple:
            self.accN = [1-self.problem.simpleQuality2(sca.listToConfiguration(p,self.problem))
                         for p in self.hof]
            self.energyN = [self.problem.simpleEnergy(sca.listToConfiguration(p,self.problem))
                            for p in self.hof]
        points = [self.evalAssignment(p, self.problem) for p in self.hof]
        if self.use_naive:
            points = [self.problem.markovQualityEnergy(sca.listToConfiguration(p,self.problem)) for p in self.hof]
        self.accM = [1-float(a) for a, e in points]
        self.energyM = [float(e) for a, e in points]
    
    def calculateReal(self):
        real = [self.problem.tester(sca.listToConfiguration(p,self.problem), special=True)
                for p in self.hof]
        self.accR = [1-float(a) for a, e in real]
        self.energyR = [float(e) for a, e in real]
    
    def calculateRealFast(self):
        real = [self.problem.tester_fast(sca.listToConfiguration(p,self.problem), special=True)
                for p in self.hof]
        self.accR = [1-float(a) for a, e in real]
        self.energyR = [float(e) for a, e in real]
    
    def getMarkov(self):
        points = [self.evalAssignment(p, self.problem) for p in self.hof]
        acc = [1-float(a) for a, e in points]
        energy = [float(e) for a, e in points]
        return acc, energy
    
    def getReal(self):
        points = [self.problem.tester_fast(sca.listToConfiguration(p,self.problem), special=True)
                for p in self.hof]
        acc = [1-float(a) for a, e in points]
        energy = [float(e) for a, e in points]
        return acc, energy
    
    def getNaive(self):
        acc = [1-self.problem.simpleQuality2(sca.listToConfiguration(p,self.problem))
                         for p in self.hof]
        energy = [self.problem.simpleEnergy(sca.listToConfiguration(p,self.problem))
                            for p in self.hof]
        return acc, energy
    
    
    def showMarkov(self):
        plt.plot(self.accM,self.energyM)
        plt.show()
    
    def show(self):
        self.fill()
        plt.plot(self.accM,self.energyM)
        plt.plot(self.accR,self.energyR)
        plt.plot(self.accN,self.energyN)
        plt.show()
        
    def show_labeled(self, x_label = "Accuracy [%]", y_label = "Energy consumption [mA]", size=None, save=None):
        self.fill()
        fig = plt.figure()
        if size:
            fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)   
        ax.plot(self.accM,self.energyM, label='Markov approx.')
        ax.plot(self.accR,self.energyR, label='Real trade-offs')
        ax.plot(self.accN,self.energyN, label='Naive approx.')
        ax.legend(loc = "upper right")
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        if save:
            fig.savefig(save+'.pdf', bbox_inches='tight')
        return fig
    
    def compute_difference(self):
        na = [abs(n-r) for (n,r) in zip(self.accN, self.accR)]
        ne = [abs(n-r) for (n,r) in zip(self.energyN, self.energyR)]
        ma = [abs(m-r) for (m,r) in zip(self.accM, self.accR)]
        me = [abs(m-r) for (m,r) in zip(self.energyM, self.energyR)]
        na = sum(na)/len(na)
        ne = sum(ne)/len(ne)
        ma = sum(ma)/len(ma)
        me = sum(me)/len(me)
        return na, ne, ma, me
    
    def testOne(self,index):
        pop = self.hof[index]
        configuration = sca.listToConfiguration(pop, self.problem)
        print (configuration.name)
        print ("Markov: "+str(map(float,self.evalAssignment(pop, self.problem))))
        print ("Naive: "+str((self.problem.simpleQuality2(configuration),
                            self.problem.simpleEnergy(configuration))))
        print ("Real: "+str(self.problem.tester(configuration, special=True)))
        
    def showStatistics(self): 
        if self.problem.track:
            print ("Number of evaluations: "+str(self.problem.numEvals))
            print ("Time for evaluation: "+str(self.problem.timeEvals))
            print ("Number of different settings: "+str(self.problem.numSettings))
            print ("Time for a single setting: "+str(self.problem.timeSettings))
        else:
            print ("Tracking disabled")
    
    def fill(self):
        if not hasattr(self, 'accN'):
            self.accN = [0]*len(self.hof)
        if not hasattr(self, 'accM'):
            self.accM = [0]*len(self.hof)
        if not hasattr(self, 'accR'):
            self.accR = [0]*len(self.hof)
        if not hasattr(self, 'energyN'):
            self.energyN = [0]*len(self.hof)
        if not hasattr(self, 'energyM'):
            self.energyM = [0]*len(self.hof)
        if not hasattr(self, 'energyR'):
            self.energyR = [0]*len(self.hof)
    
    def save(self, name): 
        f = open(name+".sol","w")
        self.fill()
        for i in range(len(self.hof)):
            f.write("{};{};{};{};{};{};{}\n".format(self.hof[i],self.accN[i],
                self.energyN[i],self.accM[i],self.energyM[i],self.accR[i],self.energyR[i]))
        f.close()        
        
    def load(self,name):
        f = open(name+".sol","r")
        tokens = [line.split(";") for line in f.readlines()] 
        self.hof = [[int(i) for i in x[0].replace(']','').replace('[','').split(",")] for x in tokens]     
        self.accN = [float(x[1]) for x in tokens]     
        self.energyN = [float(x[2]) for x in tokens]
        self.accM = [float(x[3]) for x in tokens]     
        self.energyM = [float(x[4]) for x in tokens]    
        self.accR = [float(x[5]) for x in tokens]     
        self.energyR = [float(x[6]) for x in tokens]
        f.close()


# In[1]:

def mutate_list(sequence, indpb=0.05, max_length = 0):
    if max_length==0:
        print ("Error: the index length is 0")
    for i in range(len(sequence)):
        if random.random()<indpb:
            sequence[i] = random.randint(0, max_length)
    return (sequence,)


# In[6]:

def get_generations(solver, tree = None):
    generations = {}
    current = 0
    if solver is None:
        gtree = tree
    else:
        gtree = solver.history.genealogy_tree
    #print gtree    
    for i in range(1,len(gtree)+1):
        if len(gtree[i]) == 0:
            generations[i] = 0
        else:
            m = max([generations[gtree[i][j]] for j in range(len(gtree[i]))])
            generations[i] = max(current, m+1)
            current = generations[i]
    return generations


# In[275]:

def get_ages(solver, generations, evaluator = None, history = None):
    ages = defaultdict(list)
    for subject, age in generations.items():
        if solver == None:
             ages[age]= ages[age]+[evaluator(history[subject])]
        else:
            ages[age]= ages[age]+[solver.evalAssignment(solver.history.genealogy_history[subject], solver.problem)]
    cum_array = []
    cum_ages = {}
    for key in ages.keys():
        cum_array += ages[key]
        cum_ages[key] = list(cum_array)
    return ages, cum_ages


# In[184]:




# In[18]:

def reverse_first(lst):
    return [(1-l[0],) + tuple(l[1:]) for l in lst]


# In[8]:

def hypervolume(lst, reference = (1,1)):
    volume = 0
    volume += (reference[0] - lst[0][0]) * (reference[1] - lst[0][1])
    #print lst
    #print volume
    #print lst[0]
    #print lst[1]
    for i, (a,b) in enumerate(lst[1:]):
        #print i
        #print lst[0]
        #print lst[i]
        ap, bp = lst[i]
        #print ap, a, b, bp
        #volume += (a - ap) * (reference[1] - b)
        volume += (bp - b) * (reference[0] - a)
    return volume

