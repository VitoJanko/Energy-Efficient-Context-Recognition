
# coding: utf-8

# In[1]:
from ee import simpletree as st, sharingtree as sht


class CostSensitiveTree():
    
    def __init__(self,contexts, cost_function, feature_to_sensor = None, feature_groups = None, 
                 tree_type = "default", min_samples = 1, extension = 0, default = None, weight = 1):
        if tree_type not in ["default", "pruned", "batched"]:
            raise ValueError("tree_type must be 'default', 'pruned', or 'batched'")
        self.tree_type = tree_type
        self.node = None
        self.min_samples = min_samples
        self.miscls = None
        self.contexts = contexts
        self.extension = extension
        self.default = default
        self.weight = weight
        self.feature_to_sensor = feature_to_sensor
        self.feature_groups = None
        self.original_cost_function = cost_function
        if feature_to_sensor is None and feature_groups is None:
            raise AssertionError("Either feature_to_sensor or feature_groups parameter must be set")
        if feature_to_sensor is None and feature_groups is not None:
            self.calc_feature_to_sensor(feature_groups)
        self.cost_function = self.calc_cost_function(cost_function, self.weight)
        self.calc_feature_groups(feature_to_sensor, feature_groups)
        self.calc_miscls()
    
    def copy(self):
        tree = CostSensitiveTree(contexts = self.contexts, cost_function =self.original_cost_function, 
             feature_to_sensor = self.feature_to_sensor, 
             feature_groups = self.feature_groups, 
             tree_type = self.tree_type, min_samples = self.min_samples,
             extension = self.extension, default=self.default, weight = self.weight)
        tree.node = self.node
        return tree
        
    
    def set_weight(self, weight):
        self.weight = weight
        self.cost_function = self.calc_cost_function(self.original_cost_function, self.weight)     
    
    def calc_feature_groups(self,feature_to_sensor, feature_groups):
        if feature_groups is not None and type(feature_groups) == list:
            self.feature_groups = feature_groups
        elif feature_groups is not None and type(feature_groups) == dict:
            self.feature_groups = feature_groups.values()
        else:
            sensors = list(set(feature_groups.values()))
            self.feature_groups = [[] for _ in range(len(sensors))]
            for k,v in feature_groups.items():
                self.feature_groups[sensors.index(v)].append(k)
        
                    
    def calc_feature_to_sensor(self,feature_groups):
        self.feature_to_sensor ={}
        if type(feature_groups) == dict:
            for sensor in feature_groups:
                for feature in feature_groups[sensor]:
                    self.feature_to_sensor[feature] = sensor
        if type(feature_groups) == list:
            for i,sensor in enumerate(feature_groups):
                for feature in sensor:
                    self.feature_to_sensor[feature] = i    

    def calc_cost_function(self,cost_function, weight):
        return lambda current, history, costs: weight * overall_cost(current, history, costs, 
                                                            fn = cost_function, att_to_sensor = self.feature_to_sensor)
    
    def calc_miscls(self):
        miscls_0 = [[0 if i==j else 1 for j in range(len(self.contexts))] for i in range(len(self.contexts))]
        self.miscls = {self.contexts[j] : 
                       {self.contexts[i] : c for (i,c) in enumerate(row)} for (j,row) in enumerate(miscls_0)}
    
    def fit(self,x1, y1, x_p = None, y_p = None):
        if x_p is None:
            x_p = x1
            y_p = y1
            
        self.node = st.Node()
        if self.tree_type == "default":
            self.node.create(x1,y1, None ,self.miscls, cost_function=self.cost_function, 
            min_samples=self.min_samples, extension=self.extension, default=self.default)
            
        if self.tree_type == "pruned":
            self.node.create(x1,y1, None, self.miscls, cost_function=self.cost_function, 
                min_samples=self.min_samples, extension=self.extension, default=self.default)
            self.node.prune(x_p,y_p)

        if self.tree_type == "batched":
            self.node = sht.create_cost_share(x1,y1,x_p,y_p,None,self.miscls, self.feature_groups,
                                 self.contexts, cost_function=self.cost_function, min_samples=self.min_samples, 
                                 extension = self.extension,
                                 default=self.default)     
    
    def predict(self,x2):
        return [self.node.classify(row) for index, row in x2.iterrows()]
    
    def energy(self, x, original_cost_function=None, weight = 1):
        if original_cost_function is None:
            original_cost_function = self.original_cost_function
            cost_function = self.calc_cost_function(self.original_cost_function, weight)
        else:
            cost_function = self.calc_cost_function(original_cost_function, weight)
        return self.node.energy(x,None,cost_function=cost_function) + weight*original_cost_function([])
        


# In[ ]:

def overall_cost(current, history, costs, fn = None, att_to_sensor = None):
    sensor_set = set()
    current_sensor = att_to_sensor[current]
    for attribute in history: 
        sensor_set.add(att_to_sensor[attribute])
    before = fn(sensor_set)
    sensor_set.add(current_sensor)
    after = fn(sensor_set)
    return after-before

