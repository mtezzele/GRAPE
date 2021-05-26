from grape.general_graph import GeneralGraph
from grape.fault_diagnosis import FaultDiagnosis

F = FaultDiagnosis("./input_files/switch_line.csv")
F.check_input_with_gephi()

initial = {'10': (0., 0.), '1': (0., 2.), 'A': (0., 4.),
    'S1': (2., 2.), '2': (4., 2.), 'S2': (6., 2.), '3': (8., 2.),
    '11': (8., 0.), 'S3': (10., 2.), '4': (12., 2.), 'S4': (14., 2.),
    '5': (16., 2.), '12': (16., 0.), 'B': (16., 4.), 'S5': (18., 2.),
    '6': (20., 2.), 'S6': (22., 2.), '7': (24., 2.), '13': (24., 0.),
    'C': (24., 4.), 'S7': (26., 2.), '8': (28., 2.), 'S8': (30., 2.),
    '9': (32., 2.), '14': (32., 0.)}

F.G.print_graph(initial_pos=initial, size=200, arrow_size=4, fsize=10,
    fixed_nodes=list(F.G), title='Switch line (integer)', input_cmap='Accent',
    legend_loc='upper center', legend_ncol=8, legend_anchor=(0.5, 1.05),
    legend_fsize=7)

F.simulate_element_perturbation(["1"])
print("\nPredecessors of S1: ", list(F.G.predecessors('S1')))
print("\nSuccessors of S1: ", list(F.G.successors('S1')))

F.G.print_graph(initial_pos=initial, size=200, arrow_size=4, fsize=10,
    fixed_nodes=list(F.G), title='Switch line (node 1 perturbed)',
    input_cmap='Accent', legend_loc='upper center', legend_ncol=8,
    legend_anchor=(0.5, 1.05), legend_fsize=7)

D = FaultDiagnosis("./input_files/switch_line.csv")
D.simulate_element_perturbation(["2"])

D.G.print_graph(initial_pos=initial, size=200, arrow_size=4, fsize=10,
    fixed_nodes=list(D.G), title='Switch line (node 2 perturbed)',
    input_cmap='Accent', legend_loc='upper center', legend_ncol=8,
    legend_anchor=(0.5, 1.05), legend_fsize=7)

T = FaultDiagnosis("./input_files/switch_line.csv")
T.simulate_element_perturbation(["2", "3"])

T.G.print_graph(initial_pos=initial, size=200, arrow_size=4, fsize=10,
    fixed_nodes=list(T.G), title='Switch line (nodes 2 and 3 perturbed)',
    input_cmap='Accent', legend_loc='upper center', legend_ncol=8,
    legend_anchor=(0.5, 1.05), legend_fsize=7)
