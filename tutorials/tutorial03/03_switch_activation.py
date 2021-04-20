from grape.general_graph import GeneralGraph
from grape.fault_diagnosis import FaultDiagnosis

F = FaultDiagnosis("./input_files/switch_line.csv")
F.check_input_with_gephi()
F.simulate_element_perturbation(["1"])
print("\nPredecessors of S1: ", list(F.G.predecessors('S1')))
print("\nSuccessors of S1: ", list(F.G.successors('S1')))

D = FaultDiagnosis("./input_files/switch_line.csv")
D.simulate_element_perturbation(["2"])

T = FaultDiagnosis("./input_files/switch_line.csv")
T.simulate_element_perturbation(["2", "3"])
