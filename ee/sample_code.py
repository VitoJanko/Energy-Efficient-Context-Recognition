from ee import eeoptimizer as eo
from ee import eeoptimizer as eo2
import pandas as pd

from ee.cstree import CostSensitiveTree

#Tester is the main class for all utility functions, path points to the data
#Ignore warnings at the begining
tester = eo.EnergyOptimizer(path="C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets")
tester2 = eo2.EnergyOptimizer(path="C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets")


#For each dataset, first load the data and the configuration
tester.load_data_config("SHL_data", "SHL_config")
#tester2.load_data_config("SHL_data", "SHL_config")
tester2.load_data("Gib_data")
#Optionally, load sample solutions
#sample_solutions, sample_objective = tester.load_solution("SHL_sca")

frequencies = [1,2,5,10,20,30,40,50]
y1 = pd.Series(pd.read_csv("C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets\\y1_df.csv",
                           index_col=0, names=["activity"])["activity"])
y2 = pd.Series(pd.read_csv("C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets\\y2_df.csv",
                           index_col=0, names=["activity"])["activity"])
x1 = pd.read_csv("C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets\\x1_df.csv", index_col=0)
x2 = pd.read_csv("C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets\\x2_df.csv", index_col=0)

k = 22/45.0
n = 24-5*k

def energy(setting):
    #print setting, type(setting)
    return n + setting*k


node = CostSensitiveTree(tester2.contexts,
                         lambda lst: 20 if len(lst) == 0 else energy(max(lst)),
                         feature_groups={f: [c for c in x1.columns if str(f) == c.split("_")[1]] for f in frequencies},
                         weight=0.0159420289855
                         )

tester2.add_csdt_weighted(node, x1, y1, x2, y2, weights_range=(0, 0.01), use_energy_sequence=False,
                          verbose=True, name="gib_config_cs")
print(tester2.get_energy_quality())

node.fit(x1,y1)
p2 = node.predict(x2)
print((pd.Series(p2) == y2.reset_index(drop=True)).sum() / float(len(y2)))


sample = tester.encrypt_solution([10,21,31,21,30,21,0,0])
_, s_stat = tester.find_sca_static()
print("Max eng:" + str(min(tester.setting_to_energy.values())))
print("Min eng:" + str(max(tester.setting_to_energy.values())))

print(sample)


max_sca_solution = tester2.get_min_error(encrypted=False)
print (f"Max SCA: {max_sca_solution}")
#tester.set_dca_costs(10, 20)
#tester2.set_dca_costs(10, 20)

print(tester.get_model_dca([[1,2,3,4,5,6,7,8]],active=4))
print(tester2.dca_model([1,2,3,4,5,6,7,8],active=4))

print(tester.get_real_dca([[1,2,3,4,5,6,7,8]],active=4, setting=(1,1,1,1,1)))
print(tester2.dca_real([[1,2,3,4,5,6,7,8]], active=4, setting=(1,1,1,1,1)))

print(tester.get_model_sca_dca([(max_sca_solution,[1,2,3,4,5,6,7,8]),sample,(max_sca_solution,[1,2,3,4,5,6,7,8])], active=2))
print(tester2.sca_dca_model(tradeoffs=[(max_sca_solution,[1,2,3,4,5,6,7,8]),sample,(max_sca_solution,[1,2,3,4,5,6,7,8])], active=2))

print(tester.get_real_sca_dca([(max_sca_solution,[1,2,3,4,5,6,7,8]),sample,(sample,[3,4,5,6,7,8,9,10]),
                               (max_sca_solution,[1,2,3,4,5,6,7,8])], active=2))
print(tester2.sca_dca_real(tradeoffs=[(max_sca_solution,[1,2,3,4,5,6,7,8]),sample,(sample,[3,4,5,6,7,8,9,10]),
                                      (max_sca_solution,[1,2,3,4,5,6,7,8])], active=2))

print("Alt evaluation")
print(tester.sca_model([sample,sample]))
print("Surrogate evaluation")
print(tester.get_sca_evaluation(sample))
#print(tester.get_model_sca(sample)) for array
#If needed real model can be used (accepts an array, instead of sinlge solution)
print("Real evalutation")
print(tester.get_real_sca([sample]))
print(tester.get_real_sca_alt([sample]))
print(tester2.sca_real([sample]))

#tester.get_sca_dca_tradeoffs(active=1, n_points=5, max_cycle=10)
print("SCA-DCA Tradeoffs")
h, s = tester2.find_sca_dca_tradeoffs(name="temp",active=1, n_points=5, max_cycle=10)
print(h)
print(s)

print("DCA Tradeoffs")
h, s = tester2.find_dca_tradeoffs(name="temp",active=1, setting=(1, 1, 0, 1, 1))
print(h)
print(s)

print("Tradeoffs")
h, s = tester.find_sca_tradeoffs(name="temp", solution_type="binary")
print(h)
print(s)

#Evaluate a solution (surrogate model). Output: (accuracy, energy), input varies from dataset to dataset
print("Alt evaluation")
print(tester.sca_model([sample,sample]))
print("Surrogate evaluation")
print(tester.get_sca_evaluation(sample))
#print(tester.get_model_sca(sample)) for array
#If needed real model can be used (accepts an array, instead of sinlge solution)
print("Real evalutation")
print(tester.get_real_sca([sample]))

#There are four different datasets, each with its own representation described below
#However, there is a universal format

#------
#Universal format
#------

#[x1,x2,x3,...xn]
#n = len(tester.contexts)
#xi = [0,len(tester.settings))

#Before entering this format in any function, encrypt it
#Example:
print("Number of contexts, Number of settings")
print(len(tester.contexts), len(tester.settings))
encrypted = tester.encrypt_solution([1] * 8)
#encrypted = tester.encrypt_hof(list)
print("Encrypted solution")
print(encrypted)
print("Encrypted solution evaluated")
print(tester.get_sca_evaluation(encrypted))
#tester.decrypt_solution   is the reverse operation

#Specific formats below

#------
#SHL
#------

#[x1,x_8]
#x_i = (b_1,b_5)
#b_i = 0,1
#Example:  [(1, 1, 1, 0, 0), (1, 0, 1, 1, 0), (1, 1, 0, 1, 1), (1, 0, 0, 0, 0), (0, 1, 1, 1, 1),
# (1, 0, 0, 0, 1), (0, 0, 1, 0, 0), (0, 0, 1, 0, 1)]

#------
#E-Gibalec
#------

#[x1,x_4]
#x_i in [1, 2, 5, 10, 20, 30, 40, 50]
#Example:  [2,5,5,40]

#------
# #Commodity
# #------
#
# ##IMPORTANT - different inicialization
# #tester = eo.EnergyOptimizer(path="Datasets", contexts=['Eating','Exercise','Home','Out','Sleep','Transport','Work'])
#
#
# #[x1,x_7]
# #x_i = (b_1,b_7)
# #b_i = 0,1
# #Example:  [(0, 1, 0, 1, 1, 0, 1), (1, 1, 1, 1, 0, 0, 1), (1, 0, 0, 1, 0, 0, 1), (1, 1, 0, 1, 0, 0, 1),
# #  (0, 0, 0, 1, 0, 1, 0), (0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1, 0)]

#------
#Opportunity
#------

##IMPORTANT - different inicialization
#tester = eo.EnergyOptimizer(path="Datasets", quality_metric = util.f1_from_conf)


#[x1,x_18]
#x_i in tester.settings  (example: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
