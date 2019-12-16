
# coding: utf-8

# In[8]:

import matplotlib.mlab as mlab


# In[9]:

import numpy as np
import random
import matplotlib.pyplot as plt

# ### Utility

# In[2]:

from collections import defaultdict


# In[3]:

def normalizeMatrix(matrix, rows = False):
    mtrx = np.array(matrix).astype(float)
    if rows:
        for row in np.xrange(mtrx.shape[0]):
            if np.sum(mtrx[row]) > 0:
                mtrx[row] /= np.sum(mtrx[row])
            else:
                mtrx[row][row] = 1
        return mtrx
    return mtrx/np.sum(mtrx)


# In[4]:

def fill_cf_matrix(true,predicted,l):
    cf_matrix = np.zeros((l,l))
    for (t,p) in zip(true,predicted):
        cf_matrix[int(t)][int(p)] = cf_matrix[t][p] + 1
    return cf_matrix.astype(float)


# In[5]:

def fill_trans_matrix(seq,l):
    trans_matrix = np.zeros((l,l))
    for (fst,snd) in zip(seq[:-1],seq[1:]):
        trans_matrix[int(fst)][int(snd)]+=1
    return trans_matrix.astype(float)


# In[6]:

#def predict_distribution(transitions):
#    marko = Problem(None,None,None,None,None,None,None)
#    steady = marko.get_steady_state_trans(transitions)
#    return steady


# In[7]:

def calculate_distribution(indexes,l):
    real_dist = [sum([1 for a in indexes if a==u]) for u in range(l)]
    real_norm_dist = [d /float(sum(real_dist)) for d in real_dist]
    return real_norm_dist


# In[8]:

def accuracy(distribution, matrix):
    return np.dot(np.array(distribution), np.array([matrix[i][i] for i in range(len(matrix))]))


# In[9]:

def activity_to_index(activities, sequence):
    index_map = {s:i for (i,s) in enumerate(activities)}
    indexes = [index_map[s] for s in sequence]
    return index_map, indexes


# In[10]:

def remap_matrix(matrix, activity_prev, activity_new):
    index_prev, _ = activity_to_index(activity_prev,[]) 
    index_new, _ = activity_to_index(activity_new,[])
    l = len(matrix)
    m = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            m[i][j] = matrix[index_new[activity_prev[i]]][index_new[activity_prev[j]]]
    return m


# In[11]:

def activity_length_distribution(activities, sequence):
    dist = defaultdict(list)
    current = sequence[0]
    length = 0
    for s in sequence:
        if s != current:
            dist[current].append(length)
            length = 0
            current = s
        length += 1
    return dist
            


# In[75]:

def smooth(sequence, tabu):
    result = []
    for (i,s) in enumerate(sequence[1:-1]):
        i = i+1
        if s in tabu:
            result.append(s)
        elif sequence[i-1] == sequence[i+1]: 
            result.append(sequence[i-1])
        else:
            result.append(s)
    return result


# In[1]:

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


# In[40]:

def get_steady_state_trans(m):
    #if type(m)==list:
    #    m = np.array(m)
    s = m.T-np.identity(len(m))
    e = np.linalg.eig(s)
    sols = list(map(abs,np.round(e[0], 10)))
    p = e[1][:,sols.index(min(sols))]
    steady = p/sum(p)
    steady = np.array([np.real(s) for s in steady])
    return steady


# In[47]:

def get_steady_state(p):
    if type(p)==list:
        p = np.array(p)
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)


# In[1]:

def combination_generator(values, length):
    if length==0:
        return [[]]
    else:
        solutions = []
        for v in values:
            remainders = combination_generator(values, length-1)
            solutions = solutions + [[v]+remainder for remainder in remainders]
        return solutions


# In[4]:

def random_probabilities(probs):
    p = random.random()
    for i in range(len(probs)):
        if p<sum(probs[:i+1]):
            return i


# In[6]:

def class_to_numeric(seq, activities):
    acts = list(activities)
    return [acts.index(s) for s in seq]


# In[1]:

def average_length(matrix):
    return [1.0/(1-matrix[i][i]) for i in range(len(matrix))]


# In[1]:

def pareto_dominance(lst):
    lst = list(lst)
    #st[0] = [1-x for x in lst[0]]
    #rint lst
    lst.sort(key= lambda x: x[0])
    i=1
    while i<len(lst):
        if i==0:
            continue
        if lst[i][1] >= lst[i-1][1]:
            del lst[i]
            i-= 1
        elif lst[i][0] == lst[i-1][0] and lst[i][1] < lst[i-1][1]:
            #print "huh"
            del lst[i-1]
            i-= 1
        i+=1
    return lst


def f1_single(conf,i):
    TP = conf[i][i]
    FN = sum(conf[i])-TP
    FP = sum(conf[j][i] for j in range(len(conf)))-TP
    if TP==0:
        return 0
    precision = TP/float(TP+FP)
    recall = TP/float(TP+FN)
    #print precision, recall, 2*precision*recall/(precision+recall)
    return 2*precision*recall/(precision+recall)

def f1_from_conf(conf):
    #print conf
    return sum(f1_single(conf,i) for i in range(len(conf)))/len(conf)

def draw_graph(plots, labels, xlim = None, ylim=None, name = None, reverse = False, pareto = False, 
               short = False, points = None, folder = "artificial", percentage = True, percentage_energy = False,
              scatter_indices = None, color_indices = None, text_factor = 50, ylabel = "Energy", dotted_indices = None,
              thick_indices = None, show=True , xlabel = "Classification error"):
    cmap = plt.get_cmap("tab10")
    if short:
        plt.figure(figsize=(10,10))
    else:
        plt.figure(figsize=(20,10))
    for i,(p,n) in enumerate(zip(plots,labels)):
        if len(p)==2:
            x = p[0]
            y = p[1]
        else:
            x = zip(*p)[0]
            y = zip(*p)[1]
        if reverse:
            x= [1-v for v in x]
        if pareto:
            x,y = zip(*pareto_dominance(zip(x,y)))
        if percentage:
            x = [100*v for v in x]
        if percentage_energy:
            y = [100*v for v in y]
        color = cmap(i)
        linestyle = "-"
        linewidth = None
        if not dotted_indices is None and i in dotted_indices:
            linestyle = ":"
        if not thick_indices is None and i in thick_indices:
            linewidth = 4
        if not color_indices is None:
            color = cmap(color_indices[i])
        if scatter_indices is None or (not i in scatter_indices):
            plt.step(x,y, label = n, where = "post", color = color, linestyle=linestyle, linewidth= linewidth)
        else:
            plt.scatter(x,y, label = n, color = color)
    if not points is None:
        for i, point in enumerate(points):
            color = cmap(0)
            #if i== 0:
            #    color = cmap(1)
            if percentage:
                point = list(point)
                point[0] = point[0]*100
            if percentage_energy:
                point = list(point)
                point[1] = point[1]*100
            plt.plot(point[0], point[1], 'bo', markersize=15, color = color)
            if len(point)>2:
                plt.text(point[0]+0.01*text_factor, point[1]-0.005*text_factor,point[2],fontsize=18)
    plt.xlabel(xlabel, fontsize=26)
    if percentage:
        plt.xlabel(xlabel+" [%]", fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    plt.tick_params(labelsize=18)
    if len(plots)>0:
        plt.legend(prop={'size': 24})
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)
    if not name is None:
        plt.savefig('./'+folder+'/'+name+'.pdf', bbox_inches = 'tight', pad_inches = 0.2)
    plt.show()

