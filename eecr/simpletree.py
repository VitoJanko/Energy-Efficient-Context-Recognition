
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np


# In[1]:

class Node():
    
    max_splits = 100
    
    def __init__(self, parent = None):
        self.parent = parent
        self.children = {}
        self.classify_as = None
        self.attribute = None
        self.continuous = False 
        self.split = None
    
    def create(self, X, y, test_costs, miscls_cost, columns = None, max_depth = 10, min_samples = 1, show=False, 
               delay = "", default=None, cost_function=None, history=None, extension=0):
        if columns is None:
            columns = X.columns.values
        if cost_function is None:
            cost_function = lambda current,history,costs: costs[current]
        if history is None:
            history = []
        self.classify_as, default_cost = self.majority(y, miscls_cost)
        if self.classify_as is None:
            self.classify_as = default
        best_cost = default_cost
        if extension > 0:
            #TODO: Change to something sensible
            best_cost = 100000000
        unique_classes = y.unique()
        if max_depth <= 0 or len(unique_classes)==1 or len(X)<min_samples:
            if show:
                print (delay +" --> " +str(self.classify_as))
            return
        #temp = X.copy()
        #temp["y"] = y
        #print columns
        #print default_cost
        unnamed_cost = [[miscls_cost[v][u] for v in unique_classes] for u in unique_classes]
        
        for attribute in columns:
            #print attribute
            unique = X[attribute].unique()
            #print unique
            if len(unique) < 5:
                sets = [y[X[attribute]==u] for u in unique]
                total_cost = cost_function(attribute,history,test_costs)*len(X) + sum(self.majority(s,miscls_cost)[1] for s in sets)
                #print attribute, total_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    self.attribute = attribute
            else:
                split, cost = self.possible_splits2(X[attribute], y, unique_classes, unnamed_cost)
                total_cost = cost_function(attribute, history, test_costs)*len(X) + cost
                #print attribute, total_cost
                if total_cost < best_cost and split > X[attribute].min():
                    best_cost = total_cost
                    self.attribute = attribute
                    self.split = split
        #print best_cost, extension, self.attribute
        if best_cost > default_cost:
            extension-=1
        if self.attribute:
            history.append(self.attribute)
            unique = X[self.attribute].unique()
            nc = list(columns)
            if len(unique) < 5:
                nc.remove(self.attribute)
                for u in unique:
                    if show:
                        print (delay + self.attribute +" == "+str(u))
                    self.children[u] = Node(self)
                    self.children[u].create(X[X[self.attribute]==u], y[X[self.attribute]==u], 
                                test_costs, miscls_cost, nc, max_depth-1, min_samples, show, delay+"  ",
                                            cost_function=cost_function, history = list(history), extension=extension)    
            else:
                #print "Best split : "+str(self.split)
                self.continuous = True
                if show:
                    print (delay + str(self.attribute)+ " < "+ str(self.split))
                self.children[" < "+str(self.split)] = Node(self)
                self.children[" < "+str(self.split)].create(X[X[self.attribute]<self.split], y[X[self.attribute]<self.split], 
                                test_costs, miscls_cost, nc, max_depth-1, min_samples, show, delay+"  ",
                                                            cost_function=cost_function, history = list(history),
                                                            extension=extension)  
                if show:
                    print (delay + str(self.attribute)+ " >= "+ str(self.split))
                self.children[" >= "+str(self.split)] = Node(self)
                self.children[" >= "+str(self.split)].create(X[X[self.attribute]>=self.split], y[X[self.attribute]>=self.split], 
                                test_costs, miscls_cost, nc, max_depth-1, min_samples, show, delay+"  ",
                                                             cost_function=cost_function, history = list(history),
                                                            extension=extension)  
        else:
            if show:
                print(delay +" --> " +str(self.classify_as))
                
    def prune(self, X, y):
        if len(y)==0:
            return 0
        if not self.classify_as and not self.attribute:
             raise ValueError("Tree not yet built")
        acc_base = (y == self.classify_as).sum()/float(len(y))
        if not self.attribute:
            return acc_base 
        acc = 0
        if not self.continuous: 
            for child in self.children.keys():
                relevant = X[self.attribute] == child
                acc += (relevant.sum()/float(len(y))) * self.children[child].prune(X[relevant],y[relevant])
        else:
                acc = 0
                acc +=((X[self.attribute]<(self.split)).sum() / float(len(y)))*                     self.children[" < "+str(self.split)].prune(X[X[self.attribute]<self.split],
                                                                       y[X[self.attribute]<self.split])
                acc += ((X[self.attribute]>=(self.split)).sum() / float(len(y)))*                     self.children[" >= "+str(self.split)].prune(X[X[self.attribute]>=self.split],
                                                                       y[X[self.attribute]>=self.split])
        #print acc, acc_base
        if acc < acc_base:
            #print "Boom"
            self.children = {}
            self.attribute = None
        return acc
                
            
    def possible_splits2(self, x, y, unique, costs):
        xy = pd.concat([x,y], axis=1)
        xy.sort_values(xy.columns[0], inplace = True)
        #grouped = xy.groupby(xy[xy.columns[0]])
        #main = grouped.agg(lambda x:x.value_counts().index[0])
        #splits = set(main.index[main[xy.columns[1]]!=main[xy.columns[1]].shift() ])
        splits = set(xy[xy.columns[0]][xy[xy.columns[1]]!=xy[xy.columns[1]].shift() ])
        #print len(splits)
        less = {u : 0 for u in unique}
        more = {u : (xy[xy.columns[1]]==u).sum() for u in unique}
        best_error = 0
        best_split = None
        done_x = False
        last_x = None
        #print xy
        for row in xy.itertuples():
            less[row[2]] = less[row[2]] +1
            more[row[2]] = more[row[2]] -1
            if row[1] != last_x:
                done_x = False
                last_x = row[1]
            if row[1] in splits and not done_x:
                cost_low = np.min(np.dot(costs, np.array([less[u] for u in unique])))
                cost_high = np.min(np.dot(costs, np.array([more[u] for u in unique])))
                if best_split == None or cost_low + cost_high < best_error:
                    best_split = row[1]
                    best_error = cost_low + cost_high
                done_x = True
        return best_split, best_error
        
    def possible_splits(self, x, y, split_nr):
        xy = pd.concat([x,y], axis=1)
        xy.sort_values(xy.columns[0], inplace = True)
        x = xy[xy.columns[0]]
        y = xy[xy.columns[1]]
        diff = (y != y.shift()) #& (x != x.shift())
        #diff = y == y
        #print ((x.shift()[diff]+x[diff])[1:]/2.0).unique()
        return ((x.shift()[diff]+x[diff])[1:]/2.0).unique()
    
    def majority(self, y, miscls_cost):
        #print "Majority: "+str(len(y))
        if len(y) == 0:
            return None, 0
        unique = y.unique()
        costs = [self.classfy_cost(y,c,miscls_cost) for c in unique]
        index = costs.index(min(costs))
        return unique[index], costs[index]
    
    def classfy_cost(self, y, value, miscls_cost):
        return sum(miscls_cost[c][value] for c in y)
    
    def classify(self, x):
        if not self.classify_as and not self.attribute:
             raise ValueError("Tree not yet built")
        if not self.continuous: 
            if self.attribute and x[self.attribute] in self.children:
                return self.children[x[self.attribute]].classify(x)
            else:
                return self.classify_as
        else:
            if self.attribute:
                if x[self.attribute]<self.split:
                    return self.children[" < "+str(self.split)].classify(x)
                else:
                    return self.children[" >= "+str(self.split)].classify(x)
            else:
                return self.classify_as
            
    def classify_real(self, x, tree_roots, current_tree, att_to_sensor, active_sensors, active_period=1):
        if not self.classify_as and not self.attribute:
             raise ValueError("Tree not yet built")
        if (not self.attribute) or (active_sensors[att_to_sensor[self.attribute]]<0):
            #if current_tree and self.classify_as!=current_tree:
            #    nxt_sensors = tree_roots[self.classify_as]
            if self.attribute:
                active_sensors[att_to_sensor[self.attribute]] = active_period
            for key in active_sensors.keys():
                active_sensors[key] = active_sensors[key]-1
            return self.classify_as
        if self.attribute:
            active_sensors[att_to_sensor[self.attribute]] = active_period
        if not self.continuous: 
            if x[self.attribute] in self.children:
                return self.children[x[self.attribute]].classify_real(x,tree_roots, current_tree, att_to_sensor, 
                                                                      active_sensors, active_period)
            else:
                return self.classify_as
        else:
            if x[self.attribute]<self.split:
                return self.children[" < "+str(self.split)].classify_real(x,tree_roots, current_tree, att_to_sensor, 
                                                                          active_sensors, active_period)                                                       
            else:
                return self.children[" >= "+str(self.split)].classify_real(x,tree_roots, current_tree, att_to_sensor, 
                                                                           active_sensors, active_period)
                                                                                                                   
    
        
    def show(self, delay = ""):
        if len(self.children)==0:
            print (delay +" --> " +str(self.classify_as))
        else:
            for (att,child) in self.children.items():
                if not self.continuous:
                    print (delay + str(self.attribute) +" == "+str(att))
                else:
                    print (delay + str(self.attribute) + str(att))
                child.show(delay+"  ")
                
    def show_semantic(self, att_to_sensor, delay = "", parent_sensor = ""):
        if len(self.children)==0:
            return [self.classify_as], ""
        else:
            tree_string = ""
            activity_list = []
            sensor = att_to_sensor[self.attribute]
            temp_tree_string = ""
            for (i,(att,child)) in enumerate(self.children.items()):
                activities, tree = child.show_semantic(att_to_sensor, delay+"  ", parent_sensor=sensor) 
                activity_list += activities
                temp_tree_string += tree
                #if i!= len(self.children.items())-1:
                #    temp_tree_string += "\n"
            if sensor != parent_sensor:
                tree_string += delay + str(sensor) +"\n"
                tree_string += delay + "  ---> " +  str(list(set(activity_list))) +"\n"
                tree_string += temp_tree_string 
                return [], tree_string
            else: 
                return  activity_list, temp_tree_string

                
    def accuracy(self, X, y):
        return sum([1 if self.classify(X.iloc[i])==y.iloc[i] else 0 for i in range(len(y))]) / float(len(y))
    
    def accuracy_fast(self, X, y):    
        print ("It does not work")
        if len(y)==0:
            return 0
        if not self.classify_as and not self.attribute:
             raise ValueError("Tree not yet built")
        if not self.continuous: 
            if self.attribute:
                acc = 0
                for child in self.children.keys():
                    relevant = X[self.attribute] == child
                    acc += (relevant.sum()/float(len(y))) * self.children[child].accuracy_fast(X[relevant],y[relevant])
                #extra = X[self.attribute].apply(lambda x: not x in self.children.keys())
                #acc += extra.sum() * sum([y[extra] == self.classify_as]) / float(len(y[extra]))
                #print acc
                return acc
            else:
                return (y == self.classify_as).sum()/float(len(y))
        else:
            if self.attribute:
                acc = 0
                acc +=((X[self.attribute]<(self.split)).sum() / float(len(y)))*                     self.children[" < "+str(self.split)].accuracy_fast(X[X[self.attribute]<self.split],
                                                                       y[X[self.attribute]<self.split])
                acc += ((X[self.attribute]>=(self.split)).sum() / float(len(y)))*                     self.children[" >= "+str(self.split)].accuracy_fast(X[X[self.attribute]>=self.split],
                                                                       y[X[self.attribute]>=self.split])
                #print "Float: "+str(acc)
                return acc
            else:
                return (y == self.classify_as).sum()/float(len(y))
    
    def confusion_fast(self, X, y, contexts, cfm=None):
        if cfm is None:
            cfm = np.zeros((len(contexts),len(contexts)))
        if len(y)==0:
            return 0
        if not self.classify_as and not self.attribute:
             raise ValueError("Tree not yet built")
        #print self.attribute, len(X)
        if not self.attribute:
            for real in contexts:
                cfm[contexts.index(real)][contexts.index(self.classify_as)] += (y==real).sum()
            return cfm
        if not self.continuous:
            all_relevant = pd.Series(index=X.index, data = [False]*len(X[self.attribute]))
            for child in self.children.keys():
                relevant = X[self.attribute] == child
                all_relevant = all_relevant | relevant
                self.children[child].confusion_fast(X[relevant],y[relevant], contexts, cfm)
            if (~all_relevant).sum() > 0:
                for real in contexts:
                    cfm[contexts.index(real)][contexts.index(self.classify_as)] += (y[~all_relevant]==real).sum()
            return cfm
        else:
            self.children[" < "+str(self.split)].confusion_fast(X[X[self.attribute]<self.split],
                                                                       y[X[self.attribute]<self.split], contexts, cfm)
            self.children[" >= "+str(self.split)].confusion_fast(X[X[self.attribute]>=self.split],
                                                                       y[X[self.attribute]>=self.split], contexts, cfm)
            return cfm
    
    def depth(self):
        if not self.attribute:
            return 1
        d = 0
        for child in self.children.keys():
            d = max(self.children[child].depth(),d)
            return d+1
    
    def mistake_cost(self, X,y,contexts,miscls_cost):
        cf = self.confusion_fast(X, y, contexts)
        cost = 0
        for i in range(len(cf)):
            for j in range(len(cf)):
                cost += cf[i][j] * miscls_cost[contexts[i]][contexts[j]]
        return cost
                
    def get_bottom(self, X,y):
        if not self.attribute:
            return [self],[X],[y]
        nl,xl,yl = [],[],[]
        if not self.continuous: 
            for child in self.children.keys():
                relevant = X[self.attribute] == child
                nlp,xlp,ylp = self.children[child].get_bottom(X[relevant],y[relevant])
                #print nlp,xlp,ylp
                nl+=nlp
                xl+=xlp
                yl+=ylp
        else:
            nlp1,xlp1,ylp1 = self.children[" < "+str(self.split)].get_bottom(X[X[self.attribute]<self.split],
                                                                       y[X[self.attribute]<self.split])
            nlp2,xlp2,ylp2 = self.children[" >= "+str(self.split)].get_bottom(X[X[self.attribute]>=self.split],
                                                                       y[X[self.attribute]>=self.split])
            #print nlp1,xlp1,ylp1
            nl+=nlp1+nlp2
            xl+=xlp1+xlp2
            yl+=ylp1+ylp2
        return nl,xl,yl
        
    
    def confusion_matrix(self, X, y, contexts):
        cfm = np.zeros((len(contexts),len(contexts)))
        classified = [self.classify(row) for _,row in X.iterrows()]
        for real,predicted in zip(y,classified):
            cfm[contexts.index(real)][contexts.index(predicted)]+=1
        return cfm
    
    def energy(self, X, test_costs, top=True, cost_function=None, history = None):
        if cost_function is None:
            cost_function = lambda current,history,costs: costs[current]
        if history is None:
            history = []
        if len(X)==0 or (not self.attribute):
            return 0
        #print str(test_costs[self.attribute]) +" x " +str(len(X))
        total = cost_function(self.attribute,history,test_costs) * len(X)
        history.append(self.attribute)
        if not self.continuous:
            for u in self.children.keys():
                total+= self.children[u].energy(X[X[self.attribute]==u], test_costs, False, 
                                                cost_function=cost_function, history = list(history))
        else:
            total+= self.children[" < "+str(self.split)].energy(X[X[self.attribute]<self.split], test_costs, False, 
                                                                cost_function=cost_function, history = list(history))
            total+= self.children[" >= "+str(self.split)].energy(X[X[self.attribute]>=self.split], test_costs, False, 
                                                                 cost_function=cost_function, history = list(history))
        if top:
            return total / float(len(X))
        return total
    
    def energy_cumulative(self, test_costs, return_set = False, cost_function=None):
        attributes = set()
        if not self.attribute:
            if return_set:
                return attributes
            return 0
        attributes.add(self.attribute)
        for child in self.children.values():
            #print child
            attributes.update(child.energy_cumulative(test_costs, True, cost_function = cost_function))
        if return_set:
            return attributes
        total = 0
        if cost_function is None:
            cost_function = lambda current,history,costs: costs[current]
        history = []
        for att in attributes:
            total+=cost_function(att,history,test_costs)
            history.append(att)
        return total


# In[5]:




# In[56]:

#A = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
#B = [0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1]
#y = ['c','c','c','c','c','a','a','a','b','b','b','a','a','a','a','b']


# In[43]:

#X = pd.Series(A).to_frame(name = 'A')
#X['B'] = pd.Series(B)
#y = pd.Series(y)


# In[44]:

#miscls = {'a':{'a':0,'b':1,'c':1},'b':{'a':1,'b':0,'c':1},'c':{'a':1,'b':1,'c':0}}


# In[45]:

#tests = {'A':1.0/16,'B':1.0/8}


# In[54]:

#node = Node()
#node.create(X,y,tests,miscls)


# In[50]:

#node.classify(X.iloc[0])


# In[55]:

#node.show()


# In[1]:

def named_cost(costs, acts):
    return {acts[j] : {acts[i]:c for (i,c) in enumerate(row)} for (j,row) in enumerate(costs)}


# In[1]:

def amplify_cost(costs, alpha):
    amplified = {}
    for (key, value) in costs.items():
        amplified[key] = alpha*value
    return amplified     


# ## Forest creation and manipulaiton

# In[36]:

def get_border_trees(X, Y, activities, sensor_costs, miscl_costs, shared_costs, buffer_range = 1):
    trees = {}
    for activity in activities:
        cond = pd.Series([False]*len(Y))
        for i in range(buffer_range+1):
            cond = (cond | (Y.shift(i)==activity))
        y = Y[cond]
        x = X[cond]
        node = Node()
        node.create(x,y,sensor_costs,miscl_costs,cost_function=shared_costs)
        trees[activity] = node
    return trees


# In[44]:

def forest_testing(trees,X,Y, activities,  sensor_costs, miscl_costs, shared_costs):
    current = Y.iloc[0]
    cfm = np.zeros((len(activities),len(activities)))
    energy_total = 0
    for i in range(len(Y)):
        current = trees[current].classify(X.iloc[i])
        energy_total += trees[current].energy(X.iloc[[i],:],sensor_costs,cost_function=shared_costs)
        cfm [activities.index(Y.iloc[i])][activities.index(current)]+=1
    return cfm, cfm.trace()/float(cfm.sum()), energy_total / float(len(Y))


# In[5]:

def forest_testing_single(trees,X,Y, activities, sensor_costs, shared_costs):
    #print type(trees)
    if type(trees) == list:
        accuracies = [tree.accuracy_fast(X,Y) for tree in trees]
        energies = [tree.energy(X,sensor_costs,shared_costs) for tree in trees]
        cfms = [tree.confusion_fast(X,Y,activities) for tree in trees]
        cfms_energy = [energy_activity(tree, X,Y, activities, sensor_costs, shared_costs) for tree in trees]
    if type(trees) == dict:
        accuracies = {act:tree.accuracy_fast(X,Y) for (act,tree) in trees.items()}
        energies = {act:tree.energy(X,sensor_costs,shared_costs) for (act,tree) in trees.items()}
        cfms = {act:tree.confusion_fast(X,Y,activities) for (act,tree) in trees.items()}
        cfms_energy = {act:energy_activity(tree, X,Y, activities, sensor_costs, shared_costs) for (act,tree) in trees.items()}
    return accuracies, energies, cfms, cfms_energy


# In[6]:

def energy_activity(tree, X,Y, activities, sensor_costs, shared_costs):
    return [tree.energy(X[Y==act],sensor_costs,shared_costs) for act in activities]


# In[2]:

def get_OvA_trees(X, Y, activities, sensor_costs, miscl_costs, shared_costs):
    trees = {}
    one_classes = ["Activity","Other"]
    miscls = [[0 if i==j else 1 for j in range(len(one_classes))] for i in range(len(one_classes))]
    for activity in activities:
        x_ova = X
        y_ova = Y.apply(lambda x: "Activity" if x==activity else "Other")
        node = Node()
        node.create(x_ova,y_ova,sensor_costs,named_cost(miscls, one_classes), default="Other",cost_function=shared_costs)
        trees[activity] = node
    node = Node()
    node.create(X,Y,sensor_costs,miscl_costs,cost_function=shared_costs)     
    trees["General"] = node    
    return trees


# In[9]:

def ova_testing_single(trees, X,Y, activities, sensor_costs, shared_costs):
    one_classes = ["Activity","Other"]
    accuracies = {}
    energies = {}
    cfms = {}
    cfms_energy = {}
    for activity in activities:
        x_ova = X
        y_ova = Y.apply(lambda x: "Activity" if x==activity else "Other")
        accuracies[activity] = trees[activity].accuracy_fast(x_ova,y_ova) 
        energies[activity] = trees[activity].energy(x_ova,sensor_costs,shared_costs) 
        cfms[activity] = trees[activity].confusion_fast(x_ova,y_ova,one_classes) 
        #cfms_energy[activity] = [energy_activity(tree, X,Y, activities, sensor_costs, shared_costs) for tree in trees]
    accuracies["General"] = trees["General"].accuracy_fast(X,Y) 
    energies["General"] = trees["General"].energy(X,sensor_costs,shared_costs) 
    cfms["General"] = trees["General"].confusion_fast(X,Y,activities) 
    return accuracies, energies, cfms


# In[6]:

def forest_OvA_testing(trees, X, Y, activities,  sensor_costs, miscl_costs, shared_costs):
    current = Y.iloc[0]
    cfm = np.zeros((len(activities),len(activities)))
    energy_total = 0
    for i in range(len(Y)):
        pred = trees[current].classify(X.iloc[i])
        if(pred!="Activity"):
            current = trees["General"].classify(X.iloc[i])
            energy_total += trees["General"].energy(X.iloc[[i],:],sensor_costs,cost_function=shared_costs)
        else:
            energy_total += trees[current].energy(X.iloc[[i],:],sensor_costs,cost_function=shared_costs)
        cfm [activities.index(Y.iloc[i])][activities.index(current)]+=1
    return cfm, cfm.trace()/float(cfm.sum()), energy_total / float(len(Y))

