from ee import eeoptimizer as eo

a = [1,1,1,2,2,2,3]
for name in ["hristijan","vito","mitja","tone"]:
    #  quality_metric = util.f1_from_conf
    tester = eo.EnergyOptimizer(path = "Datasets")
    tester.load_data_csv("Com_data_"+name+".csv")
    #tester.load_data("Gib_data")
    tester.save_data("Com_data_"+name)

    tester.load_config_csv("Com_config_"+name)
    #tester.load_config("Com_config")
    tester.save_config("Com_config_"+name)

#print(tester.setting_to_sequence.keys())


#print(tester.sequence)
#print(tester.settings)
#print(tester.setting_to_sequence)
#print(tester.setting_to_energy)
#print(tester.setting_to_energy_sequence)

#config = [(1, 1, 1, 0, 0),(1, 1, 1, 0, 0),(1, 1, 1, 0, 0),(1, 1, 1, 0, 0),
#          (1, 1, 1, 0, 0),(1, 1, 1, 0, 0),(1, 1, 1, 0, 0),(1, 1, 1, 0, 0)]

#print(tester.get_sca_format())
#print(tester.get_sca_evaluation(config))



#print(tester.get_dca_evaluation([1,2,1,2,1,2,1,2]))

#h_sca, s_sca = tester.load_solution("Gib_sca")
#print(s_sca)

#print(tester.get_real_sca(h_sca))