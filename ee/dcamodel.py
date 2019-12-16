
# coding: utf-8

# In[16]:

import numpy as np

# In[17]:

import pickle


# In[12]:

from sklearn.metrics import confusion_matrix


# In[14]:

import pandas as pd
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap.tools import History

# In[38]:
from ee import eeutility as util


# ## Introduction

# Assuming Markov distribution of classes, the goal is to compute the ideal duty-cycle length for each class. Scheme modeled is as follows: one instance is classified, than based of the duty legth, next few instances are skipped. This is modelled with and without classification errors.  

# ### Case 0: Testing

# In[122]:

def silly_estimate(p,length):
    avg = 1/p
    correct = avg - (length-1)/2.0
    return correct/avg


# In[85]:

def simplest_estimate(p,length):
    q = 1-p
    correct = 1 + 0.5*(length-1) + 0.5*(length-1)*(q**length)
    #correct = 0.5*length + 0.5*length*q**length
    return correct/length


# In[8]:

def simple_estimate(p,length):
    q = 1-p
    correct = (1-q**length) / float(1-q) 
    return correct/length


# ### Case 1: all duty cycles of the same length, no missclasifications

# Duty cycle of length $n$: Every $n$-th instance is sampled <br />
# Note: Duty cycle length includes the original instance, length of 1 means all instances are classified <br />
# Idea:<br />
# 1.) Distribution of sampled instances is the same as the original (use Markov steady state) <br />
# 2.) Probability of error for each step $0...n$ can be calculated (use few Markov steps)<br />
# 

# In[29]:

def reduce_opposite(fn, start, n_times=3):
    for _ in range(n_times):
        start = fn(start)
        yield start


# In[33]:

def predict_duty_simple(transition, length):
    #Distribution of sampled instances
    distribution =util.get_steady_state_trans(transition)
    l = len(transition)
    #1-error made after each activity
    correct = []
    for i in range(l):
        start = np.eye(i+1,l)[i]
        #Probability for activities, for each step after duty cycle starts, if it started at activity i 
        probs = [start] + list(reduce_opposite(lambda x: np.dot(x,transition), start, length-1))
        correct.append(sum(p[i] for p in probs)/len(probs))
    #Final result = (1-error) for activity, weighted by its distribution
    return np.dot(distribution,correct)


# In[15]:

def test_duty(seq, length):
    dutied = [seq[i-i%length] for i in range(len(seq))]
    correct = sum(1 for (a,b) in zip(seq,dutied) if a==b)/float(len(seq))
    return (dutied,correct)


# ### Case 2: duty cycles of different lengths, no missclasifications

# Goal: predict accuracy, precision, recall, when each activity has a different duty-cycle length <br />
# Idea: <br />
# 1.) Classified instances have a different distribution as all instances <br />
# 1a.) Calculate the probability that the activity $a$ is classified after the activity $b$ for all $a,b$ (Markov steps)<br />
# 1b.) Calculate the classified activity distribution (Markov steady state)<br />
# 2.) Each activity generates it's own block after it is classified (classified instance and duty instances that follow)  <br />
# 2a.) For each block we know its length, how often it occures, and an expected number of times an activitiy is going to appear in it <br />
# 2b.) It's a matter of simple arithmetic to calculate desired accuracy, precision, recall using that info

# In[25]:

def predict_duty(transition, lengths):
   # marko = Problem(None,None,None,None,None,None,None)
    l = len(transition)
    ## lasts[i][j] = Probability that last actvity after duty cycle of an activity i is activity j
    lasts = []   
    ## acts[i][j] = Average probability that j-th activity appears during duty-cycle of activity i 
    acts = []
    for i in range(l):
        start = np.eye(i+1,l)[i]
        #Probability for activities, for each step after duty cycle starts, if it started at activity i 
        probs = [start] + list(reduce_opposite(lambda x: np.dot(x,transition), start, lengths[i]))
        lasts.append(probs[-1])
        acts.append([sum([a[j] for a in probs[:-1]])/lengths[i] for j in range(l)])
#Distribution of classified activities
    distributions = util.get_steady_state_trans(np.array(lasts))
    
#Precision
    precisions = [p[i] for (i,p) in enumerate(acts)]
#Recall
    #Number of activity b appearances is proportional (normalized by the total number of blocks) to 
    #(blocks of type a * length of this block * chance that an element of that block is b) for each activity a
    act_represented = [sum([s*l*a[i] for (s,l,a) in zip(distributions,lengths,acts)]) for i in range(len(lengths))]
    #Same, but only counting activities correctly detected
    act_delivered = [s*l*a for (s,l,a) in zip(distributions,lengths,precisions)]
    recalls = [d/float(r) for (r,d) in zip(act_represented, act_delivered)] 
#Accuracy
    #Total normalized number of instances
    samples = sum([a*c for (a,c) in zip(distributions,lengths)])
    #Total normalized number of corrrectly classified instances
    correct = sum(act_delivered)
    accuracy = correct/float(samples)
    
    return accuracy, precisions, recalls
    


# Testing method, different duty cycles, optional missclasification

# In[53]:

def test_duty_diverse_confusion(seq, length, confusion = None):
    if confusion is None:
        confusion = [[1 if i==j else 0 for j in range(len(length))] for i in range(len(length))]
    timer = 0
    currrent = 0
    dutied = []
    starters = []
    mistakes = [(0,0) for i in length]
    pre_rec = [[0,0,0] for i in length]
    for i in range(len(seq)):
        if timer==0:
            current = util.random_probabilities(confusion[seq[i]])
            starters.append(seq[i])
            timer = length[current]-1
        else:
            timer-=1
        pre_rec[seq[i]][0] += current==seq[i]
        pre_rec[seq[i]][1] += 1
        pre_rec[current][2] += 1
        mistakes[current]  = (mistakes[current][0]+(current==seq[i]), mistakes[current][1]+1) 
        dutied.append(current)
    correct = sum(1 for (a,b) in zip(seq,dutied) if a==b)/float(len(seq))
    precisions = [0 if float(pre_rec[i][2])==0 else pre_rec[i][0]/ float(pre_rec[i][2]) for i in range(len(pre_rec))]
    recalls = [ 0 if float(pre_rec[i][1])== 0 else pre_rec[i][0]/ float(pre_rec[i][1]) for i in range(len(pre_rec))]
    #for i in range(len(pre_rec)):
    #    precision = 0 if float(pre_rec[i][2])==0 else pre_rec[i][0]/ float(pre_rec[i][2])
    #   recall = 0 if float(pre_rec[i][1])== 0 else pre_rec[i][0]/ float(pre_rec[i][1])
    #   print
    #    print "Activity: " +str(i)
    #    print "Precision: "+ str(precision)
    #    print "Recall: " + str(recall)
    #print
    #for i in range(len(mistakes)):
    #    print mistakes[i][0]/ float(mistakes[i][1])
    #print
    #for i in range(len(length)):
    #    print sum([1 for s in starters if s==i])/float(len(starters))
    return (dutied,correct, precisions, recalls)


# ### Case 3: duty cycles of different lengths, missclasifications

# In[51]:

def predict_duty_confusion(transition, lengths, confusion=None):
    if confusion is None:
        confusion = [[1 if i==j else 0 for j in range(len(lengths))] for i in range(len(lengths))]
    #marko = Problem(None,None,None,None,None,None,None)

    l = len(transition)
    correct_ratio = []
    all_probs = {}
    

    ## acts[i][j] = Average probability that j-th activity appears during duty-cycle of activity i 
    acts = []
    
    avg_length = [0]*l
    for i in range(l):
        start = np.eye(i+1,l)[i]
        #Calculate probabilities of max length, any length could be used for any activitiy due to missclasifications 
        probs =  [start] + list(reduce_opposite(lambda x: np.dot(x,transition), start, max(lengths)))
        #Store relevant lengths
        all_probs[i] = np.array([probs[cycle] for cycle in lengths])
        #Sum up to relevant lengths
        sums = np.array([[sum([a[j] for a in probs[:lng]])/lng for lng in lengths] for j in range(l)])
        #Average number of correctly classified instances, for each activity that could be classified
        correct = [sum(a[j] for a in probs[:lengths[j]]) for j in range(l)]
        #Expected number of correctly classified instances, given missclasification rate
        total_correct = np.dot(correct,confusion[i])
        #Average length of a duty cycle of activity i depends on how is this activity recognized 
        total = np.dot(lengths,confusion[i])
        avg_length[i] = total
        #Proportion of time we are being correct for activity i
        correct_ratio.append(total_correct / float(total))
        #Average number of activities in a duty block, depends on confusion matrix
        acts.append(np.dot(sums,np.array(confusion[i]).T))
#Probability of next clasified activity
    next_starter_prob = [np.dot(all_probs[act].T, confusion[act]).T for act in range(l)]
#Distribution of classified activities
    distribution = util.get_steady_state_trans(np.array(next_starter_prob))
#Precision
    precisions = [np.dot(p.T,confusion[i]) for (i,p) in enumerate(acts)]
#Accuracy
    correct = sum([a*b*c for (a,b,c) in zip(distribution,correct_ratio,avg_length)])
    samples = sum([a*c for (a,c) in zip(distribution,avg_length)])
#Recall
    #represented = [a*c*d for (a,c,d) in zip(distribution,avg_length,acts)]
    #recognized = [sum([a*c*d[j] for (a,c,d) in zip(distribution,length,confusion,)]) for j in len(l)]
    return correct/float(samples), precisions


# In[224]:

def predict_duty_confusion_alt(transition, lengths, confusion=None, active = 1):
    if confusion is None:
        confusion = [[1 if i==j else 0 for j in range(len(lengths))] for i in range(len(lengths))]
    confusion = np.array(confusion)
    #marko = Problem(None,None,None,None,None,None,None)
    l = len(transition)
    ##probability of getting in a (true,pred) state. Indexes are flattened
    next_state_prob = []
    ##expected activity[i,j,k] = number of activity k expected in cycle generated if true activity = i, predicted = j 
    expected_sleeping = np.zeros((l,l,l))
    expected_onduty = np.zeros((l,l,l))
    
    for true in range(l):
        for pred in range(l):
            start = np.eye(true+1,l)[true]
            probs =  np.array([start] + list(reduce_opposite(lambda x: np.dot(x,transition), start, active+lengths[pred]-1)))
            probs_confused = [np.array([a*c for (a,c) in zip(p,confusion)]) for p in probs]
            next_state_prob.append(probs_confused[-1].flatten())
            expected_onduty[true][pred] = (np.sum(probs[lengths[pred]:-1], axis=0))
            expected_sleeping[true][pred] = (np.sum(probs[:lengths[pred]], axis=0))
            
    distribution = util.get_steady_state_trans(np.array(next_state_prob)).reshape((l,l))
    
    matrix = np.zeros((l,l))
    for true in range(l):
        for pred in range(l):
            matrix[true][pred] = sum([distribution[i][pred]*expected_sleeping[i][pred][true] for i in range(l)])
            matrix[true][pred] += sum([distribution[i//l][i%l]*expected_onduty[i//l][i%l][true]*confusion[true][pred]
                                       for i in range(l*l)])
    matrix/=np.sum(matrix)
    
    precision = [get_precision(matrix,i) for i in range(l)]
    recall = [get_recall(matrix,i) for i in range(l)]
    accuracy = sum([matrix[i][i] for i in range(l)])
    
    spared = np.dot(np.sum(distribution.reshape((l,l)),axis = 0),lengths)
    spared = (spared-1)/(active)
    
    return accuracy, precision, recall, matrix, spared
    


# In[5]:

def predict_duty_sca(transition, lengths, energy_costs, confusion=None, active = 1):
    if confusion is None:
        confusion = [[1 if i==j else 0 for j in range(len(lengths))] for i in range(len(lengths))]
    confusion = np.array(confusion)
    #marko = Problem(None,None,None,None,None,None,None)
    l = len(transition)
    ##probability of getting in a (true,pred) state. Indexes are flattened
    next_state_prob = []
    ##expected activity[i,j,k] = number of activity k expected in cycle generated if true activity = i, predicted = j 
    expected_sleeping = np.zeros((l,l,l))
    expected_onduty = np.zeros((l,l,l))
    
    for true in range(l):
        for pred in range(l):
            start = np.eye(true+1,l)[true]
            probs =  np.array([start] + list(reduce_opposite(lambda x: np.dot(x,transition), start, active+lengths[pred]-1)))
            probs_confused = [np.array([a*c for (a,c) in zip(p,confusion)]) for p in probs]
            next_state_prob.append(probs_confused[-1].flatten())
            expected_onduty[true][pred] = (np.sum(probs[lengths[pred]:-1], axis=0))
            expected_sleeping[true][pred] = (np.sum(probs[:lengths[pred]], axis=0))
            
    distribution = util.get_steady_state_trans(np.array(next_state_prob)).reshape((l,l))
    
    matrix = np.zeros((l,l))
    for true in range(l):
        for pred in range(l):
            matrix[true][pred] = sum([distribution[i][pred]*expected_sleeping[i][pred][true] for i in range(l)])
            matrix[true][pred] += sum([distribution[i//l][i%l]*expected_onduty[i//l][i%l][true]*confusion[true][pred]
                                       for i in range(l*l)])
    matrix/=np.sum(matrix)
    
    precision = [get_precision(matrix,i) for i in range(l)]
    recall = [get_recall(matrix,i) for i in range(l)]
    accuracy = sum([matrix[i][i] for i in range(l)])
    
    #print "Energy", energy_costs
    #print "Dist", np.sum(distribution.reshape((l,l)),axis = 0)
    avg_energy = np.dot(np.sum(distribution.reshape((l,l)),axis = 0),energy_costs)
    #print avg_energy
    spared = np.dot(np.sum(distribution.reshape((l,l)),axis = 0),lengths)
    spared = (spared-1)/(active)
    
    return accuracy, precision, recall, matrix, spared, avg_energy
    


# In[ ]:

def precalculate(transition, max_length, confusion=None,active=1, lengths = None):
    if confusion is None:
        confusion = [[1 if i==j else 0 for j in range(len(transition))] for i in range(len(transition))]
    l = len(transition)
    next_state_prob = []
    expected_sleeping_all = {}
    expected_active_all = {}
    probs_all = []
    for true in range(l):
        start = np.eye(true+1,l)[true]
        probs =  np.array([start] + list(reduce_opposite(lambda x: np.dot(x,transition), start, active+max_length-1)))
        probs_all.append(probs)
        expected_sleeping = np.zeros(l).T
        for i in range(max_length):
            #print expected_sleeping
            expected_sleeping += probs[i]
            expected_sleeping_all[(true,i)] = np.array(expected_sleeping)
            expected_active = np.sum(probs[i+1:i+active], axis=0)
            expected_active_all[(true,i)] = np.array(expected_active)
    return probs_all, expected_sleeping_all, expected_active_all      


# In[11]:

def duty_predict_fast(transition, lengths, prob, sleeping_exp, working_exp, confusion=None, active = 1, energy_costs=None):
    #print 1, int(round(time.time() * 1000))
    #global flag
    if confusion is None:
        confusion = [[1 if i==j else 0 for j in range(len(lengths))] for i in range(len(lengths))]
    confusion = np.array(confusion)
    #marko = Problem(None,None,None,None,None,None,None)
    l = len(transition)
    next_state_prob = []
    #print 2, int(round(time.time() * 1000))     
    for t_from in range(l):
        for p_from in range(l):
            probs = prob[t_from][lengths[p_from]+active-1]
            next_states = np.multiply(confusion.T,probs).T
            next_state_prob.append(next_states.flatten())
    #print 3, int(round(time.time() * 1000))       
    #if flag:
    #    print confusion
    #    
    #    print transition
    distribution = util.get_steady_state_trans(np.array(next_state_prob)).reshape((l,l))
    #if flag:
    #    print "---"
    #    print distribution
    #print 4, int(round(time.time() * 1000))
    matrix = np.zeros((l,l))
    for true in range(l):
        for pred in range(l):
            matrix[true][pred] = sum([distribution[i][pred]*sleeping_exp[(i,lengths[pred]-1)][true] for i in range(l)])
    if active>1:
        for cycle_true in range(l):
            for cycle_pred in range(l):
                matrix += distribution[cycle_true][cycle_pred]*                         np.multiply(confusion.T,working_exp[(cycle_true,lengths[cycle_pred]-1)]).T
           # matrix[true][pred] += sum([distribution[i/l][i%l]*working[(i/l,lengths[i%l]-1)][true]*confusion[true][pred] 
           #                            for i in range(l*l)])
    #print 5, int(round(time.time() * 1000))
    matrix/=np.sum(matrix)
    
    #precision = [get_precision(matrix,i) for i in range(l)]
    #recall = [get_recall(matrix,i) for i in range(l)]
    accuracy = sum([matrix[i][i] for i in range(l)])
    avg_energy = 0
    if energy_costs is not None:
        if type(energy_costs[0])==list:
            avg_energy = np.sum(np.multiply(distribution.reshape((l,l)), np.array(energy_costs).T))
        else:
            avg_energy = np.dot(np.sum(distribution.reshape((l,l)),axis = 0),energy_costs)
    spared = np.dot(np.sum(distribution.reshape((l,l)),axis = 0),lengths)
    spared = (spared-1)/(active)
   
    #if flag:
    #    print active
    #    print accuracy
    #    print spared
    #    flag = False
    #print 6, int(round(time.time() * 1000))
    return accuracy, spared, matrix, avg_energy


# ### Intermezzo, final definitions

# In[8]:

def duty_prediction(transition, lengths, confusion=None, active = 1):
    if type(lengths) == int:
        lengths = [lengths]*len(transition)
    if not confusion is None:
        confusion = util.normalizeMatrix(confusion, rows = True)
    accuracy, precision, recall, matrix, spared = predict_duty_confusion_alt(transition, lengths, confusion, active)
    return accuracy, spared, matrix


# In[1]:

def duty_prediction_sca(transition, lengths, energies, confusion=None, active = 1):
    if type(lengths) == int:
        lengths = [lengths]*len(transition)
    if not confusion is None:
        confusion = util.normalizeMatrix(confusion, rows = True)
    accuracy, precision, recall, matrix, spared,  avg_energy= predict_duty_sca(transition, lengths, energies, confusion, active)
    return accuracy, spared, matrix, avg_energy


# In[9]:

def duty_test(seq, lengths, confusion = None, active = 1):
    if type(lengths) == int:
        lengths = [lengths]*len(pd.Series(seq).unique())
    if not confusion is None:
        confusion = util.normalizeMatrix(confusion, rows = True)
        
    #active = active -1
    seq, correct, precisions, recalls, spared =  test_duty_diverse_confusion_active(seq, lengths, confusion, active)
    return correct, spared, precisions, recalls


# In[23]:

def duty_energy(length, base_off, base_on):
    prop_on = 1 / float(length)
    prop_off = (length-1) / float(length)
    return prop_off*base_off + prop_on*base_on


# ### Case 4: duty cycles of different lengths, missclasifications

# In[77]:

def f1_single(conf,i):
    precision = get_precision(conf,i)
    recall = get_recall(conf,i)
    #print precision, recall, 2*precision*recall/(precision+recall) 
    return 2*precision*recall/(precision+recall)

def f1_from_conf(conf):
    #print conf
    return sum(f1_single(conf,i) for i in range(len(conf)))/len(conf)


# In[44]:

def get_precision(conf,i):
    TP = conf[i][i]
    FP = sum(conf[j][i] for j in range(len(conf)))-TP
    if TP==0:
        return 0
    return TP/float(TP+FP)

def get_recall(conf,i):
    TP = conf[i][i]
    FN = sum(conf[i])-TP
    if TP==0:
        return 0
    return TP/float(TP+FN)


# In[148]:

def predict_duty_active(transition, lengths, active):
    #marko = Problem(None,None,None,None,None,None,None)
    l = len(transition)
    ## lasts[i][j] = Probability that last actvity after duty cycle of an activity i is activity j
    lasts = []   
    ## lasts[i][j] = Probability that last actvity after duty cycle of an activity i is activity j
    nexts = []   
    ## acts[i][j] = Average probability that j-th activity appears during duty-cycle of activity i 
    acts = []
    for i in range(l):
        start = np.eye(i+1,l)[i]
        #Probability for activities, for each step after duty cycle starts, if it started at activity i 
        probs = [start] + list(reduce_opposite(lambda x: np.dot(x,transition), start, active + lengths[i]))
        lasts.append(probs[-1])
        nexts.append([sum([a[j] for a in probs[lengths[i]:-1]])/max(1,active) for j in range(l)])
        acts.append([sum([a[j] for a in probs[:lengths[i]]])/lengths[i] for j in range(l)])
    #Distribution of classified activities
    distributions = util.get_steady_state_trans(np.array(lasts))
    matrix = np.zeros((l,l))
    for i in range(l):
        for j in range(l):            
            matrix[i][j] += acts[j][i]*distributions[j]*lengths[j]
            if i==j:
                matrix[i][j] += sum([d*n[i]*active for (d,n) in zip(distributions,nexts)])
    matrix/=np.sum(matrix)
    precision = [get_precision(matrix,i) for i in range(l)]
    recall = [get_recall(matrix,i) for i in range(l)]
    accuracy = sum([matrix[i][i] for i in range(l)])
    spared = np.dot(distributions,lengths)
    spared = (spared-1)/(active+1)
    return accuracy, precision, recall, spared, matrix


# In[77]:

#a = [1,0,0,0,0,1,1,1,1,0]
#a1 = [1,2,1,2,1,2,1,2,1,2]
#a1 = [1,0,0,0,0,1,1,1,1,0]
#a2 = [1,0,0,1,0,1,1,1,1,0]
#test_sca_dca(a,[a1,a2],[1,3],[1,1],0,lambda x: x, 1)


# In[79]:

def test_sca_dca(sequence_true, sequences, lengths, energies, energy_off, performance, active=1, energy_sequences=None):
    #print lengths
    #print energies
    #print active
    #print energy_off
    #print [len(x) for x in SHL_configuration]
    #active = max(0,active-1)
    active_timer = active
    sleep_timer = -1
    sequence = []
    current_activity = sequence_true[0]
    energy = 0
    for i in range(len(sequence_true)):
        #print "Step 1"
        #print active_timer, sleep_timer
        if active_timer>0:
            current_activity = sequences[current_activity][i]
            sequence.append(current_activity)
            if energy_sequences is None:
                energy += energies[current_activity]
            else:
                energy += energy_sequences[current_activity][i]
            #print "append 1"
            active_timer-=1
            if active_timer==0:
                sleep_timer = lengths[current_activity]-1
                if sleep_timer == 0:
                    active_timer = active
            
            
        #print "Step 2"
        #print active_timer, sleep_timer
        elif sleep_timer > 0:
            sequence.append(current_activity)
            energy += energy_off
            sleep_timer-=1 
            if sleep_timer == 0:
                active_timer = active
        #print "Step 3"
        #print active_timer, sleep_timer
        
    #print sequence
    cf = confusion_matrix(sequence_true, sequence) 
    prf = performance(cf)
    eng = energy / float(len(sequence_true))
    
    return prf, eng  #cf


# In[10]:

def test_duty_diverse_confusion_active(seq, length, confusion = None, active=0):
    active = max(0,active-1)
    if confusion is None:
        confusion = [[1 if i==j else 0 for j in range(len(length))] for i in range(len(length))]
    timer = 0
    currrent = 0
    dutied = []
    starters = []
    mistakes = [(0,0) for i in length]
    pre_rec = [[0,0,0] for i in length]
    is_on = active
    dutied_count = 0
    nondutied_count = 0
    l = len(length)
    matrix = [[0]*l for i in range(l)]
    for i in range(len(seq)):
        if timer==0:
            nondutied_count += 1
            #print i
            #print seq[i]
            #print confusion
            #print confusion[seq[i]]
            current = util.random_probabilities(confusion[seq[i]])
            starters.append(seq[i])
            if is_on == 0:
                is_on = active
                timer = length[current]-1
            else:
                is_on-=1
        else:
            dutied_count +=1
            timer-=1
        pre_rec[seq[i]][0] += current==seq[i]
        pre_rec[seq[i]][1] += 1
        pre_rec[current][2] += 1
        matrix[seq[i]][current] +=1
        mistakes[current]  = (mistakes[current][0]+(current==seq[i]), mistakes[current][1]+1) 
        dutied.append(current)
    correct = sum(1 for (a,b) in zip(seq,dutied) if a==b)/float(len(seq))
    precisions = [0 if float(pre_rec[i][2])==0 else pre_rec[i][0]/ float(pre_rec[i][2]) for i in range(len(pre_rec))]
    recalls = [ 0 if float(pre_rec[i][1])== 0 else pre_rec[i][0]/ float(pre_rec[i][1]) for i in range(len(pre_rec))]
    matrix = np.array(matrix).astype(float)
    matrix/=np.sum(matrix)
    #print matrix
    
    #for i in range(len(pre_rec)):
    #    precision = 0 if float(pre_rec[i][2])==0 else pre_rec[i][0]/ float(pre_rec[i][2])
    #   recall = 0 if float(pre_rec[i][1])== 0 else pre_rec[i][0]/ float(pre_rec[i][1])
    #   print
    #    print "Activity: " +str(i)
    #    print "Precision: "+ str(precision)
    #    print "Recall: " + str(recall)
    #print
    #for i in range(len(mistakes)):
    #    print mistakes[i][0]/ float(mistakes[i][1])
    #print
    #for i in range(len(length)):
    #    print sum([1 for s in starters if s==i])/float(len(starters))
    return (dutied,correct, precisions, recalls, dutied_count/float(nondutied_count))


# ### Finding solutions - Genetic

# In[22]:

import random
def mutate(sequence, indpb=0.05, length = 30):
    for i in range(len(sequence)):
        if random.random()<indpb:
            sequence[i] = random.randint(1, length)
    return (sequence,)


# In[17]:

def evalAssignment(x):
    result = predict_duty_active(transition, x, 0)
    return (result[0],result[3])


# In[6]:

def geneticSolutions(length, maxCycle, evaluator, NGEN = 200, gen_size = 200, indpb=0.05, seeded=False, save_history=False,
                    verbose = False):
    creator.create("Fitness", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 1, maxCycle)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=indpb, length = maxCycle)
    toolbox.register("select", tools.selNSGA2)
    if save_history:
        history = History()
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)
    
    track = False
    MU = gen_size
    LAMBDA = gen_size*2
    CXPB = 0.7
    MUTPB = 0.2
    pop = toolbox.population(n=MU)
    if seeded:
        for i in range(length):
            pop[0][i] = 1
            pop[1][i] = maxCycle
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,halloffame=hof, verbose=verbose)
    if save_history:
        return hof, history.genealogy_tree, history.genealogy_history
    return hof


# In[19]:

def calculateHof(hof):
    return [evalAssignment(x) for x in hof]


# In[7]:

def evalDutyAssignment(x, transitions, confusion):
    result = duty_prediction(transitions, x, confusion = confusion)
    return (result[0],result[1])


# In[11]:

def reverse_and_normalize_duty(solutions, mn=0, mx=1):
    acc = [1-x for x in zip(*solutions)[0]]
    eng = [duty_energy(x+1, mn, mx) for x in zip(*solutions)[1]]
    return (acc,eng)


# In[2]:

def duty_save_hof(name, hof):
    dbfile = open(name, 'ab') 
    pickle.dump(hof, dbfile)                      
    dbfile.close() 

def duty_load_hof(name): 
    dbfile = open(name, 'rb')      
    hof = pickle.load(dbfile) 
    dbfile.close() 
    return hof

