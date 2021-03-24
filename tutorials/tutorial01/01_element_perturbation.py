from grape.general_graph import GeneralGraph
from grape.fault_diagnosis import FaultDiagnosis

F = FaultDiagnosis("./input_files/TOY_graph.csv")

F.check_input_with_gephi()
F.simulate_element_perturbation(["1"])
