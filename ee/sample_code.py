from ee import eeoptimizer as eo
import pandas as pd

from ee.cstree import CostSensitiveTree

#Tester is the main class for all utility functions, path points to the data
#Ignore warnings at the begining
tester = eo.EnergyOptimizer(path="C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets")



#For each dataset, first load the data and the configuration
#tester.load_data_config("Gib_data", "gib_config_cs")
tester.load_data("Gib_data")
#tester2.load_data_config("SHL_data", "SHL_config")
#tester2.load_data("Gib_data")
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


node = CostSensitiveTree(tester.contexts,
                         lambda lst: 20 if len(lst) == 0 else energy(max(lst)),
                         feature_groups={f: [c for c in x1.columns if str(f) == c.split("_")[1]] for f in frequencies},
                         weight=0.0159420289855
                         )

tester.add_csdt_borders(node, x1, y1, x2, y2, buffer_range=5, weights_range=(0, 0.01), use_energy_sequence=True,
                        verbose=True, name="gib_config_cs_border")

tester = eo.EnergyOptimizer(path="C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets")
tester.load_data("Gib_data")

tester.add_csdt_weighted(node, x1, y1, x2, y2, weights_range=(0, 0.01), use_energy_sequence=True,
                         verbose=True, name="gib_config_cs")

print(tester.get_energy_quality())

for pair in tester.find_sca_static()[1]:
    print(pair)

print("CS-SCA-DCA")
tester.set_dca_costs(20, 46)
h_sca_dca_cs, s_sca_dca_cs = tester.find_sca_dca_tradeoffs(cstree=True, name="sca_dca_cs_real",n_points=3,
                                                           max_cycle=30, active=2, verbose=True)
print(s_sca_dca_cs)
#print("CS-SCA")
#h_sca_cs, s_sca_cs = tester.find_sca_tradeoffs(cstree=True, name="sca_cs_real")
#print(s_sca_cs)
#print("CS-SCA-WRONG")
#h_sca_cs2, s_sca_cs2 = tester.find_sca_tradeoffs(cstree=True, name="sca_cs")
#p2 = tester.setting_to_sequence[0.009333333333333332]
#print((pd.Series(p2) == y2.reset_index(drop=True)).sum() / float(len(y2)))



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
