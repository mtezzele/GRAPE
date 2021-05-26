from grape.general_graph import GeneralGraph
from grape.fault_diagnosis import FaultDiagnosis

F = FaultDiagnosis("./input_files/TOY_graph.csv")

initial = { '11': (2., 2.), '19': (6., 2.), '12': (10., 2.),
    '14': (6., 10.), '13': (10., 10.), '18': (2., 10.), '5': (-2., 2.),
    '3': (-4., 6.), '1': (-6., 10.), '2': (-8., 6.), '4': (-10., 2.),
    '6': (-6., -2.), '7': (-10, -10.), '8': (-2, -10.), '10': (2., -2.),
    '17': (10., -2.), '16': (10., -10.), '9': (2., -10.), '15': (6., -6.)}

F.G.print_graph(initial_pos=initial, size=800, edge_width=3., arrow_size=7,
    fsize=12, fixed_nodes=list(F.G), title='TOY graph (integer)',
    input_cmap='Accent', legend_loc='upper center', legend_ncol=4,
    legend_anchor=(0.5, 1.2), legend_fsize=12)

F.check_input_with_gephi()
F.simulate_element_perturbation(["1"])

F.G.print_graph(initial_pos=initial, size=800, edge_width=3., arrow_size=7,
    fsize=12, fixed_nodes=list(F.G), title='TOY graph (perturbed)',
    input_cmap='Accent', legend_loc='upper center', legend_ncol=4,
    legend_anchor=(0.5, 1.2), legend_fsize=12)
