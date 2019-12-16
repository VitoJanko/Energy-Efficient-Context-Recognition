from ee import eeoptimizer as eo

#Tester is the main class for all utility functions, path points to the data
#Ignore warnings at the begining
tester = eo.EnergyOptimizer(path="Datasets")

#For each dataset, first load the data and the configuration
tester.load_data_config("SHL_data", "SHL_config")
#Optionally, load sample solutions
sample_solutions, sample_objective = tester.load_solution("SHL_sca")
sample = [1,1,1,1]
_, s_stat = tester.get_sca_static()
print("Max eng:" + str(min(tester.setting_to_energy.values())))
print("Min eng:" + str(max(tester.setting_to_energy.values())))



#Evaluate a solution (surrogate model). Output: (accuracy, energy), input varies from dataset to dataset
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
