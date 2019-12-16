
# coding: utf-8

# In[4]:

import numpy as np
import collections
import time

# In[5]:

import deap


# In[24]:

class Configuration:
    name = ""
    quality = None
    energy = None
    settings = None
    problem = None
    
    def __init__(self, problem, setting_names):
        self.problem = problem
        self.settings = [self.create_setting(i,s,problem) for i,s in enumerate(setting_names)]
        self.name = problem.make_name(self)
    
    def create_setting(self, index, name, problem):
        if problem.supply_index:
            name = (index,name)
        if str(name) in problem.setting_pool: 
            #print "Loaded!"
            s = problem.setting_pool[str(name)].copy()
            s.index = index
            return s
        else: 
            #print "Made!"
            if problem.track: 
                problem.numSettings+=1
                t0 = time.clock()  
            s =  Subsetting(name, index, problem.estimate_matrix(name))
            problem.setting_pool[str(name)] = s
            if problem.track:
                self.problem.timeSettings+=time.clock()-t0
            return s
    
    def get_confusion_matrix(self, activity):
        return self.settings[activity].matrix
    
    def get_confusion_matrix_norm(self, activity):
        return self.settings[activity].matrix_norm
    
    def get_setting(self, activity):
        return self.settings[activity]


# In[8]:

class Subsetting:
    def __init__(self, name, index, matrix):
            self.name = name
            self.index = index
            self.matrix = matrix
            self.matrix_norm = [list(map(lambda x: 0 if sum(b)==0 else float(x)/sum(b), b)) for b in matrix]
            
    def copy(self):
        return Subsetting(self.name,self.index,self.matrix)






def get_steady_state_trans(m):
    s = m.T-np.identity(len(m))
    e = np.linalg.eig(s)
    sols = list(map(abs,np.round(e[0], 10)))
    p = e[1][:,sols.index(min(sols))]
    steady = p/sum(p)
    steady = np.array([np.real(s) for s in steady])
    return steady


# In[2]:

class Problem: 
    estimate_energy = None 
    estimate_quality = None
    estimate_matrix = None
    activities = None
    proportions = None
    transitions = None
    make_name = None
    setting_pool = None
    track = False
    supply_index = False
    energy_by_dist = False
    energy_by_matrix = False
    
    def __init__(self, activities, estimate_matrix, estimate_quality, estimate_energy, 
                 proportions, transitions, namer = None, local_quality = None): 
        self.activities = activities
        self.estimate_energy = estimate_energy
        self.estimate_quality = estimate_quality
        self.estimate_matrix = estimate_matrix
        self.proportions = np.array(proportions)
        self.transitions = transitions
        self.setting_pool = {}
        self.make_name = namer or self.default_name
        self.estimate_local_quality = local_quality or self.default_name
    
    def default_name(self, configuration): 
        n = [str(self.activities[i])+ ": "+str(configuration.settings[i].name) 
             for i in range(len(self.activities))]
        return "\n".join([str(x) for x in n]) 
    
    def default_accuracy(setting):
        return setting.matrix_norm[setting.index][setting.index]
    
    def get_markov_transitions(self, setting):
        length = len(self.activities)
        lenSquare = len(self.activities)**2
        chain = [[0]*lenSquare for i in range(lenSquare)]
        for i in range(lenSquare):
            for j in range(lenSquare):
                fromR, toR, fromP, toP = i//length,j//length,i%length,j%length
                probA = self.transitions[fromR][toR]
                #print "Woah a new print: "+str(len(setting.get_confusion_matrix_norm(fromP)))+" "+str(toR)+" "+str(toP) 
                probB = setting.get_confusion_matrix_norm(fromP)[toR][toP]
                #print i, j, probA, probB, probA*probB
                chain[i][j] = probA*probB
        #print chain
        return np.array(chain)
        
    def get_steady_state(self, setting):
        m = self.get_markov_transitions(setting)
        steady = self.get_steady_state_trans(m)
        #print steady
        return np.reshape(steady, (int(len(steady)**0.5), int(len(steady)**0.5)))
    
    def get_steady_state_trans(self,m):
        s = m.T-np.identity(len(m))
        e = np.linalg.eig(s)
        sols = list(map(abs,np.round(e[0], 10)))
        p = e[1][:,sols.index(min(sols))]
        steady = p/sum(p)
        steady = np.array([np.real(s) for s in steady])
        return steady
    
    def simpleQuality(self, configuration):
        #print map(self.estimate_local_quality, configuration.settings)
        return np.dot(list(map(self.estimate_local_quality, configuration.settings)),self.proportions.T)
    
    def simpleQuality2(self, configuration):
        conf = [self.proportions[i]*np.array(s.matrix_norm[i]) for i,s in enumerate(configuration.settings)]
        return self.estimate_quality(conf)
    
    def simpleEnergy(self, configuration):
        if self.energy_by_dist:
            return self.estimate_energy(self.proportions.T, configuration)
        if self.energy_by_matrix:
            return np.dot(np.diagonal(list(map(self.estimate_energy, configuration.settings))),self.proportions.T)
        return np.dot(list(map(self.estimate_energy, configuration.settings)),self.proportions.T)
    
    def markovQualityEnergy(self, configuration):
        steady = self.get_steady_state(configuration)
        return self.markovQuality(configuration, steady = steady),                 self.markovEnergy(configuration, steady = steady)
    
    def markovQuality(self, configuration, steady = None):
        if steady is None: 
            steady = self.get_steady_state(configuration)
        if not (steady>-0.00001).all():
            return -float("inf")
        #print steady
        return self.estimate_quality(steady)
    
    def markovEnergy(self, configuration, steady = None):
        if steady is None: 
            steady = self.get_steady_state(configuration)
        if not (steady>-0.00001).all():
            return float("inf")
        #print steady
        steady_sum = np.sum(list(map(lambda x: np.real(x),steady)), axis=0).T
        if self.energy_by_dist:
            return self.estimate_energy(steady_sum, configuration)
        if self.energy_by_matrix:
            return np.sum(np.multiply(steady, np.array(list(map(self.estimate_energy, configuration.settings))).T))
        return np.dot(list(map(self.estimate_energy, configuration.settings)),steady_sum)
    
    def save(self, name): 
        f = open(name+".set","w")
        for (name,setting) in self.setting_pool.items():
            conf = setting.matrix
            l = len(conf)
            f.write(name+";"+";".join([str(conf[i][j]) for i in range(l) for j in range(l)])+"\n")
        f.close()
        
    def load(self,name):
        f = open(name+".set","r")
        for line in f.readlines(): 
            chars = line.split(";")
            name = chars[0]
            conf = chars[1:]
            l = int(len(conf)**0.5)
            restored = [[int(s) for s in conf[i*l:(i+1)*l]] for i in range(l)]
            setting = [int(x) for x in name.replace(']','').replace('[','').replace(',','').split(" ")]
            self.setting_pool[name] = Subsetting(setting, 0, restored)
        f.close()
        
    def setup_real(self, create=None, modify=None, test=None, x1=None, y1=None, x2=None, y2=None, create_test=None):
        self.create = create
        self.modify = modify
        self.test = test
        self.create_test = create_test
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        
    def tester(self,configuration, verbose=False, labels = None, special=False,
              full_output = False):
        labels = labels or self.activities
        classifiers = [self.create(s,self.x1,self.y1) for s in configuration.settings]
        quality, energy = 0, 0 
        predicted = 0
        conf = [[0]*len(self.activities) for i in range(len(self.activities))]
        output = []
        energy_bucket = collections.defaultdict(int)
        for i in range(len(self.x2)):
            inst, cls = self.x2.iloc[i], self.y2.iloc[i]
            predicted = self.activities.index(self.test(classifiers[predicted],self.modify(configuration.settings[predicted],inst)))
            if self.energy_by_dist:
                energy_bucket[configuration.settings[predicted]] += 1/float(len(self.x2)) 
            else:
                energy += self.estimate_energy(configuration.settings[predicted])
            #if not cls in activities:
            #    cls = activities[0]
            if verbose: 
                print (labels[self.activities.index(cls)], labels[predicted])
            if full_output:
                output+= [(labels[self.activities.index(cls)], labels[predicted])]
            quality += self.activities.index(cls)==predicted
            conf[self.activities.index(cls)][predicted]+=1
        energy = energy/float(len(self.x2))
        if self.energy_by_dist:
            distribution = [energy_bucket[x] for x in configuration.settings]
            energy = self.estimate_energy(distribution,configuration)
        quality = quality/float(len(self.x2))
        if special:
            quality = self.estimate_quality(conf)
        if full_output:
            return quality, energy, output 
        return quality, energy 
    
    def tester_fast(self,configuration, verbose=False, labels = None, special=False,
              full_output = False):
        labels = labels or self.activities
        classified = [self.create_test(s,self.x1,self.y1,self.x2,self.y2) for s in configuration.settings]
        quality, energy = 0, 0 
        predicted = 0
        conf = [[0]*len(self.activities) for i in range(len(self.activities))]
        output = []
        energy_bucket = collections.defaultdict(int)
        for i in range(len(self.x2)):
            inst, cls = self.x2.iloc[i], self.y2.iloc[i]
            #print classified[0]
            predicted = self.activities.index(classified[predicted][i])
            if self.energy_by_dist:
                energy_bucket[configuration.settings[predicted]] += 1/float(len(self.x2)) 
            else:
                energy += self.estimate_energy(configuration.settings[predicted])
            #if not cls in activities:
            #    cls = activities[0]
            if verbose: 
                print (labels[self.activities.index(cls)], labels[predicted])
            if full_output:
                output+= [(labels[self.activities.index(cls)], labels[predicted])]
            quality += self.activities.index(cls)==predicted
            conf[self.activities.index(cls)][predicted]+=1
        energy = energy/float(len(self.x2))
        if self.energy_by_dist:
            distribution = [energy_bucket[x] for x in configuration.settings]
            energy = self.estimate_energy(distribution,configuration)
        quality = quality/float(len(self.x2))
        if special:
            quality = self.estimate_quality(conf)
        if full_output:
            return quality, energy, output 
        return quality, energy 
    


# In[4]:

class Problem_transition: 
    estimate_energy = None 
    estimate_quality = None
    estimate_matrix = None
    activities = None
    proportions = None
    transitions = None
    make_name = None
    setting_pool = None
    track = False
    supply_index = True
    
    def __init__(self, activities, estimate_matrix, estimate_quality, estimate_energy, proportions,
                 transitions, namer = None): 
        self.activities = activities+["Transition"]
        self.estimate_energy = estimate_energy
        self.estimate_quality = estimate_quality
        self.estimate_matrix = estimate_matrix
        self.proportions = np.array(proportions)
        self.transitions = transitions
        self.setting_pool = {}
        self.make_name = namer or self.default_name
    
    def default_name(self, configuration): 
        n = [str(self.activities[i])+ ": "+str(configuration.settings[i].name) 
             for i in range(len(self.activities))]
        return "\n".join([str(x) for x in n]) 
    
    def default_accuracy(setting):
        return setting.matrix_norm[setting.index][setting.index]
    
    def get_markov_transitions(self, config):
        length = len(self.activities)-1
        lengthP = length+1
        lenSquare = length*lengthP
        chain = [[0]*lenSquare for i in range(lenSquare)]
        for i in range(lenSquare):
            for j in range(lenSquare):
                fromR, toR, fromP, toP = i//lengthP,j//lengthP,i%lengthP,j%lengthP
                probA = self.transitions[fromR][toR]
                probB = 0
                same = 1 -(fromP==toR)
                if fromP == length and toP != length: 
                    #print fromP, toR, toP
                    probB = config.get_confusion_matrix_norm(fromP)[toR][toP]
                if fromP != length and toP == length: 
                    probB = config.get_confusion_matrix_norm(fromP)[same][1]
                if fromP == toP and toP != length: 
                    probB = config.get_confusion_matrix_norm(fromP)[same][0]
                #print i, j, probA, probB, same
                #print fromR, toR, fromP, toP
                chain[i][j] = probA*probB
        #print chain
        return np.array(chain)
        
    def get_steady_state(self, setting):
        m = self.get_markov_transitions(setting)
        s = m.T-np.identity(len(m))
        e = np.linalg.eig(s)
        sols = list(map(abs,np.round(e[0], 10)))
        p = e[1][:,sols.index(min(sols))]
        steady = p/sum(p)
        #print steady
        return np.reshape(steady, (int(len(steady)**0.5), -1))
    
    
    def simpleQuality2(self, configuration):
        ##TO DO 
        return 0
    
    def simpleEnergy(self, configuration):
        ##TO DO 
        return 0
    
    def markovQualityEnergy(self, configuration):
        steady = self.get_steady_state(configuration)
        return self.markovQuality(configuration, steady = steady),                 self.markovEnergy(configuration, steady = steady)
    
    def markovQuality(self, configuration, steady = None):
        if steady is None: 
            steady = self.get_steady_state(configuration)
        return self.estimate_quality(steady)
    
    def markovEnergy(self, configuration, steady = None):
        if steady is None: 
            steady = self.get_steady_state(configuration)
        return np.dot(list(map(self.estimate_energy, configuration.settings)),np.sum(steady, axis=0))
    
    def save(self, name): 
        f = open(name+".set","w")
        for (name,setting) in self.setting_pool.items():
            conf = setting.matrix
            l = len(conf)
            f.write(name+";"+";".join([str(conf[i][j]) for i in range(l) for j in range(l)])+"\n")
        f.close()
        
    def load(self,name):
        f = open(name+".set","r")
        for line in f.readlines(): 
            chars = line.split(";")
            name = chars[0]
            conf = chars[1:]
            l = int(len(conf)**0.5)
            restored = [[int(s) for s in conf[i*l:(i+1)*l]] for i in range(l)]
            setting = tuple([int(x) for x in name.replace(')','').replace('(','').replace(',','').split(",")])
            self.setting_pool[name] = Subsetting(setting, 0, restored)
        f.close()
        
    def setup_real(self, create, modify, test, create_t, modify_t, test_t,
                   x1, y1, x2, y2):
        self.create = create
        self.modify = modify
        self.test = test
        self.create_t = create_t
        self.modify_t = modify_t
        self.test_t = test_t
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
     
    
    def tester(self,configuration, verbose=False, labels = None, special=False,
              full_output = False):
        labels = labels or self.activities
        classifiers = [self.create(s,self.x1,self.y1) for s in configuration.settings[:-1]]
        classifier_trans = self.create_t(configuration.settings[-1],self.x1,self.y1)
        quality, energy = 0, 0 
        predicted = 0
        conf = [[0]*(len(self.activities)) for i in range(len(self.activities)-1)]
        output = []
        for i in range(len(self.x2)):
            inst, cls = self.x2.iloc[i], self.y2.iloc[i]
            if predicted != len(self.activities)-1:
                energy += self.estimate_energy(configuration.settings[predicted])
                p = self.test(classifiers[predicted],
                            self.modify(configuration.settings[predicted],inst))
                if p==1:
                    predicted = len(self.activities)-1
            else:
                energy += self.estimate_energy(configuration.settings[-1])
                predicted = self.activities.index(self.test_t(classifier_trans,
                        self.modify_t(configuration.settings[-1],inst)))
            if verbose: 
                print (labels[self.activities.index(cls)], labels[predicted])
            if full_output:
                output+= [(labels[self.activities.index(cls)], labels[predicted])]
            quality += self.activities.index(cls)==predicted
            conf[self.activities.index(cls)][predicted]+=1
        energy = energy/float(len(self.x2))
        quality = quality/float(len(self.x2))
        if special:
            quality = self.estimate_quality(conf)
        if full_output:
            return quality, energy, output 
        return quality, energy 
    
    


# In[1]:

def get_steady_state(m):
    m = np.array(m)
    s = m.T-np.identity(len(m))
    e = np.linalg.eig(s)
    sols = list(map(abs,np.round(e[0], 10)))
    p = e[1][:,sols.index(min(sols))]
    steady = p/sum(p)
    steady = np.array([np.real(s) for s in steady])
    return steady


# In[12]:

def get_transitions(stream, activities, count_self=True):
    transitions_real = np.zeros((len(activities),len(activities)))
    for i in range(len(activities)):
        for j in range(len(activities)):
            if i!=j or count_self:
                transitions_real[i][j] = len(stream[(stream.shift()==activities[i]) & (stream==activities[j])])
    sums = transitions_real.sum(axis=1)
    sums[sums==0] = 1
    transitions_real =(transitions_real.T/sums).T
    return transitions_real


# In[1]:

def remember(func):
    results = {}
    def wrapper(setting,*args,**kargs):
        if setting in results:
            return results[setting]
        else:
            results[setting] = func(setting,*args,**kargs)
            return results[setting] 
    return wrapper


def unpack(func):
    def wrapper(setting, *args,**kargs):
        #print setting, type(setting)
        if isinstance(setting, Subsetting):
            setting = setting.name
        if isinstance(setting, str):
            return func(setting, *args,**kargs)
        #if isinstance(setting, deap.creator.Individual):
            
        if type(setting)!=int and type(setting)!=np.int32:
            setting = tuple(setting)
            #if type(setting) == list or type(setting)==numpy.ndarray:
            #    setting = tuple(setting)
            if len(setting)==1:
                setting = setting[0]
        return func(setting, *args,**kargs)
    return wrapper


# In[13]:

def accuracy_matrix(matrix): 
    return sum([matrix[i][i] for i in range(len(matrix))])/float(np.real(sum([sum(m) for m in matrix])))


# In[2]:

def listToConfiguration(lst,problem):
    settings = np.reshape(lst,(len(problem.activities),-1))
    settings = [(s if len(s)>1 else s[0]) for s in settings]
    #print settings
    return Configuration(problem, settings)   

