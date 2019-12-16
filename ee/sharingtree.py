
# coding: utf-8

# In[1]:

from ee import simpletree as st


# In[1]:

def amplify_cost(costs, alpha):
    if costs is None:
        return costs
    amplified = {}
    for (key, value) in costs.items():
        amplified[key] = alpha*value
    return amplified     


# In[1]:

def create_cost_share(X, y, X_prune, y_prune, test_costs, miscls_cost, feature_sets, contexts, max_depth = 20, min_samples = 1, show=False, 
           delay = "", default=None, cost_function=None, history=None, extension=0, root_node=None, historic_features = None):
    #print "Feature sets: "+str(len(feature_sets))
    #print "Shape"
    #print X.shape
    #print y.shape
    
    if history is None:
        history = []
    
    if historic_features is None:
        historic_features = []
    
    if root_node is None:
        root_node = st.Node()
        root_node.create(X, y, amplify_cost(test_costs,0), miscls_cost, max_depth = 0)

    classify_as, default_cost = root_node.majority(y, miscls_cost)
    best_node = None
    best_cost = default_cost
    best_set = None

    #print default_cost
    for fset in feature_sets:
        if len(fset) == 0:
            continue
        fset_mitja = fset+historic_features
        fset_mitja = list(set(fset_mitja) & set(X.columns))
        #print fset_mitja
        node = st.Node()
        node.create(X, y, amplify_cost(test_costs,0), miscls_cost, max_depth = max_depth, min_samples = min_samples, 
                    show=show, delay = delay, default=default, cost_function=cost_function, history=list(history), 
                    extension=extension, columns = fset_mitja)
        #node.show()
        test_cost = cost_function(fset[0],history,test_costs)*len(X)
        #print fset[0],history,test_costs, len(X), test_cost
        mistake_cost = node.mistake_cost(X,y,contexts,miscls_cost)
        total_cost = test_cost+mistake_cost
        #print total_cost, test_cost, mistake_cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = node
            best_set = fset

    if not best_node is None:
        #print "PRE-PRUNE"
        #best_node.show()
        depth = best_node.depth()
        history = list(history)
        history.append(best_set[0])
        feature_sets = list(feature_sets)
        feature_sets.remove(best_set)
        historic_features = list(historic_features)
        historic_features += best_set

        root_node.parent =  best_node.parent
        root_node.children = best_node.children
        root_node.classify_as = best_node.classify_as
        root_node.attribute = best_node.attribute
        root_node.continuous = best_node.continuous
        root_node.split = best_node.split
        #print "POST-PRUNE"
        root_node.prune(X_prune,y_prune)
        #root_node.show()
        
        bottom_nodes, bottom_x, bottom_y = root_node.get_bottom(X,y)
        _, X_prune_b, y_prune_b = root_node.get_bottom(X_prune,y_prune)
        #print "Bottom nodes: "+str(len(bottom_nodes))
        for bottom_node, Xb,yb, Xpb, ypb in zip(bottom_nodes, bottom_x, bottom_y, X_prune_b, y_prune_b):
            if len(yb)>=min_samples:
                create_cost_share(Xb, yb, Xpb, ypb,test_costs, miscls_cost, feature_sets,  contexts, 
                              max_depth = max(1,max_depth-depth), min_samples = min_samples, show=show, delay = delay, 
                              default=default, cost_function=cost_function, history=history, extension=extension, 
                              root_node=bottom_node, historic_features=historic_features)
    
    return root_node

