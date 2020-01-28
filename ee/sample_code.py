from ee import eeoptimizer as eo
from ee import eeutility as util
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ee.cstree import CostSensitiveTree

activities = ["still","still", "walk", "walk","run","run","car","car","subway","subway"]
gps_velocity = [0,1,0,1,0,1,66,68,66,68]
acc_motion = [0,1,4,5,4,5,0,1,0,1]
acc_period = [5,4,5,4,10,11,5,4,5,4]
mag_field = [48,49,48,49,48,49,48,49,102,203]
data = pd.DataFrame()
data["gps_velocity"] = gps_velocity
data["acc_motion"] = acc_motion
data["acc_period"] = acc_period
data["mag_field"] = mag_field
data["activity"] = activities
x = data.drop(["activity"], axis=1)
y = data["activity"]
contexts = y.unique()

classifier = DecisionTreeClassifier()
optimizer = eo.EnergyOptimizer(y)

feature_groups = {"acc": ["acc_motion", "acc_period"], "gps": ["gps_velocity"], "mag":["mag_field"]}

def sensor_costs(setting):
    print(setting)
    cost = 20
    if setting[0] == 1:
        cost += 10
    if setting[1] == 1:
        cost+=30
    if setting[2] == 1:
        cost+=15
    return cost

optimizer.add_subsets(x,y,x,y,classifier, feature_groups=feature_groups, n = 3,
                     setting_to_energy = sensor_costs)


#Tester is the main class for all utility functions, path points to the data
#Ignore warnings at the begining
classified = {}
classified["a"] = [0,0,0,0,  2,1,2,1,  2,2,2,0]
classified["b"] = [0,0,0,0,  1,1,1,2,  0,1,2,2]
classified["c"] = [1,1,1,1,  1,1,1,1,  1,1,1,1]
classified["d"] = [0,0,0,0,  1,1,1,1,  2,2,2,2]
classified["e"] = [0,0,0,0,  0,0,0,0,  2,2,2,2]
sequence = [0,0,0,0,  1,1,1,1,  2,2,2,2]
energy = {"a":2, "b":2, "c":1, "d":3, "e":1}
eo.EnergyOptimizer(sequence=sequence, setting_to_sequence=classified, setting_to_energy=energy)
tester = eo.EnergyOptimizer(path="C:\\Users\\vito\\Desktop\\EnergyEfficient\\Datasets")



#For each dataset, first load the data and the configuration
#tester.load_data_config("Gib_data", "gib_config_cs")
#tester.load_data("Gib_data")
tester.load_data_config("SHL_data", "SHL_config")
#tester2.load_data("Gib_data")
#Optionally, load sample solutions
#sample_solutions, sample_objective = tester.load_solution("SHL_sca")

h2, s2, = tester.find_sca_tradeoffs(name="sca_test")

#h4, s4 = tester.load_solution("dca_test")
max_setting = sorted(tester.quality().items(), key=lambda x: -x[1])[0][0]

h4, s4 = tester.find_dca_tradeoffs(name="dca_test", max_cycle=30, setting=max_setting)
h5, s5 = tester.find_aimd_tradeoffs(name="aimd_test")
print(h5)
print(s5)
util.draw_tradeoffs([s4, s5], ["dca", "episodic"], scatter_indices=[])

h3, s3, = tester.find_simple_tradeoffs(name="simple_test")
print(h3)
print(s3)
h, s = tester.load_solution("coh_test")

#util.draw_tradeoffs([s, s3],["coh", "simple"], scatter_indices=[1])

h, s, = tester.find_coh_tradeoffs(name="coh_test")
h2, s2, = tester.find_sca_tradeoffs(name="sca_test")
print(s)
print(s2)
s2r = tester.sca_real(h2)
print(s2r)
util.draw_tradeoffs([s, s2r],["coh","sca"])


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
