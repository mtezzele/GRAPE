"""FaultDiagnosis module"""

import logging
import sys
import warnings
import networkx as nx
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools

from .general_graph import GeneralGraph
from .parallel_general_graph import ParallelGeneralGraph

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class FaultDiagnosis():
    """Class FaultDiagnosis.

    Perturbation of a GeneralGraph object.
    Perturbation can be simulated on a list of elements, or areas.
    From one element, the perturbation propagates in all directions,
    unless an isolating element is present.
    """

    def __init__(self, filename, parallel=False):
        """

        Create an input graph, with the structure contained in the input file.

        :param str filename: input file in CSV format.
        :param parallel: flag for parallel graph creation,
            default to False.
        :type parallel: bool, optional
        """

        if parallel:
            self.G = ParallelGeneralGraph()
        else:
            self.G = GeneralGraph()

        self.df, self.edges_df = self.G.load(filename)

        self.damaged_areas = set()

    def check_input_with_gephi(self):
        """

        Write list of nodes and list of edges CSV format files,
        to visualize the input with Gephi.
        """

        gephi_nodes_df = self.df.reset_index()
        gephi_nodes_df.rename(columns={'index': 'mark'}, inplace=True)

        fields = [ 'mark', 'description', 'init_status',
                   'perturbation_resistant', 'area' ]

        gephi_nodes_df[fields].to_csv('check_import_nodes.csv', index=False)

        orphans = self.edges_df['father_mark'].str.contains('NULL')
        self.edges_df = self.edges_df[~orphans]
        self.edges_df.to_csv('check_import_edges.csv', index=False)

    def check_before(self):
        """

        Describe the topology of the integer graph, before the
        occurrence of any perturbation in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.
        """

        original_source_user_paths = []

        measure_fields = ['nodal_efficiency', 'local_efficiency', 'service']
        self.update_output(measure_fields, prefix='original_')

        for source in self.G.sources:
            for user in self.G.users:
                if nx.has_path(self.G, source, user):

                    osip = list(nx.all_simple_paths(self.G, source, user))
                    oshp = self.G.shortest_path[source][user]
                    oshpl = self.G.shortest_path_length[source][user]
                    oeff = 1 / oshpl
                    ids = source + user

                else:
                    oshpl = 'NO_PATH'
                    osip = 'NO_PATH'
                    oshp = 'NO_PATH'
                    oeff = 'NO_PATH'
                    ids = source + user

                original_source_user_paths.append({
                  'from':
                  source,
                  'to':
                  user,
                  'original_shortest_path_length':
                  oshpl,
                  'original_shortest_path':
                  oshp,
                  'original_simple_path':
                  osip,
                  'original_pair_efficiency':
                  oeff,
                  'ids':
                  ids
               })

        self.paths_df = pd.DataFrame(original_source_user_paths)

        # In check before we include also centralities
        centrality_fields = ['closeness_centrality', 'betweenness_centrality',
            'indegree_centrality', 'outdegree_centrality', 'degree_centrality']
        self.update_output(centrality_fields)

    def fitness_evaluation(self, individual, perturbed_nodes,
        initial_condition):
        """

        Evaluation of fitness on individual.
        The individual is a list of conditions for the graph switches
        (True or False).
        Edges connecting its predecessors are removed if the switch state
        is set to 'False'.

        :param list perturbed_nodes: nodes(s) involved in the
            perturbing event.
        :param dict initial_condition: initial status (boolean) for the graph
            switches.
        """

        n_actions = np.sum(np.not_equal(list(initial_condition.values()),
            individual))

        T = GeneralGraph()
        T.add_nodes_from(self.G) 
        for (u, v, d) in self.G.edges(data=True):
            T.add_edge(u, v, father_condition=d['father_condition'],
                weight=d['weight'])
        nx.set_node_attributes(T, self.G.initial_service,
            name='initial_service')
        nx.set_node_attributes(T, self.G.perturbation_resistant,
            name='perturbation_resistant')
        nx.set_node_attributes(T, self.G.type, name='type')

        for switch, opened in zip(initial_condition.keys(), individual):
            if not opened:
                for pred in list(T.predecessors(switch)):
                    T.remove_edge(pred, switch)

        for node in perturbed_nodes:
            if node in T.nodes():

                _, broken_nodes = self.rm_nodes(node, T)
                broken_nodes = list(set(broken_nodes))

                for n in broken_nodes: T.remove_node(n)

        return (n_actions - sum(T.service.values())- len(T),)

    def optimizer(self, perturbed_nodes, initial_condition, params):
        """

        Genetic algorithm to optimize switches conditions, using DEAP.

        :param list perturbed_nodes: nodes(s) involved in the
            perturbing event.
        :param dict initial_condition: initial status (boolean) for the graph
            switches.
        :param dict params: values for the optimizer evolutionary algorithm.
            Dict of: {str: int, str: int, str: float, str: float, str: int}.

            'npop' -- number of individuals for each population (default to 300)
            'ngen' -- total number of generations (default to 100)
            'indpb' -- independent probability for attributes to be changed
                (default to 0.6)
            'tresh' -- threshold for applying crossover/mutation
                (default to 0.5)
            'nsel' -- number of individuals to select (default to 5)
        """

        logging.getLogger().setLevel(logging.INFO)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
        toolbox = base.Toolbox()
        # Attribute generator
        toolbox.register("attribute_bool", random.choice, [True, False])
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual,
            toolbox.attribute_bool, len(self.G.switches))
        toolbox.register("population", tools.initRepeat, list,
            toolbox.individual)
    
        toolbox.register("evaluate", self.fitness_evaluation)
        toolbox.register("mate", tools.cxUniform, indpb=params['indpb'])
        toolbox.register("mutate", tools.mutShuffleIndexes,
            indpb=params['indpb'])
        toolbox.register("select", tools.selBest)

        pop = toolbox.population(n=params['npop'])
        # Evaluate the entire population
        fitnesses = [toolbox.evaluate(ind, perturbed_nodes,
            initial_condition) for ind in pop]

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
    
        fitnesses = fits
        result = []
        # Begin the evolution

        while g < params['ngen']:
            # A new generation
            g = g + 1

            # Select the next generation individuals
            offspring = toolbox.select(pop, params['nsel'])
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < params['tresh']:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
    
            for mutant in offspring:
                if random.random() < params['tresh']:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [toolbox.evaluate(ind, perturbed_nodes,
                initial_condition) for ind in invalid_ind]
    
            for ind, fit in zip(invalid_ind, list(fitnesses)):
                ind.fitness.values = fit

            pop[:] = offspring[:] + invalid_ind[:]

            best = toolbox.select(pop, 1)
            result.append([best[0], best[0].fitness.values[0]])

        logging.getLogger().setLevel(logging.DEBUG)

        return result

    def check_after(self):
        """

        Describe the topology of the potentially perturbed graph,
        after the occurrence of a perturbation in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.
        """

        final_source_user_paths = []

        measure_fields = ['nodal_efficiency', 'local_efficiency', 'service']
        self.update_output(measure_fields, prefix='final_')

        for source in self.G.sources:
            for user in self.G.users:
                if nx.has_path(self.G, source, user):

                    sip = list(nx.all_simple_paths(self.G, source, user))
                    shp = self.G.shortest_path[source][user]
                    shpl = self.G.shortest_path_length[source][user]
                    neff = 1 / shpl
                    ids = source + user

                else:

                    shpl = 'NO_PATH'
                    sip = 'NO_PATH'
                    shp = 'NO_PATH'
                    neff = 'NO_PATH'
                    ids = source + user

                final_source_user_paths.append({
                    'from': source,
                    'area': self.G.area[source],
                    'to': user,
                    'final_shortest_path_length': shpl,
                    'final_shortest_path': shp,
                    'final_simple_path': sip,
                    'final_pair_efficiency': neff,
                    'ids': ids
                })

        if final_source_user_paths:
            final_df = pd.DataFrame(final_source_user_paths)
            self.paths_df = pd.merge(self.paths_df, final_df,
                on=['from', 'to', 'ids'], how='outer')

    def rm_nodes(self, node, graph, visited=None, broken_nodes=None):
        """

        Remove nodes from the graph in a depth first search way to
        propagate the perturbation.
        Nodes are not deleted if perturbation resistant.
        Moreover, valves are not deleted if encountered
        during the propagation of a the perturbation.
        They are deleted, instead, if object of node deletion themselves.

        :param str node: the id of the node to remove.
        :param nx.DiGraph graph: graph on which to apply the node deletion
        :param visited: nodes already visited, default to None.
        :type visited: set, optional
        :param broken_nodes: nodes that got broken along the perturbation,
            default to None.
        :type visited: list, optional
        """

        if visited is None:
            visited = set()
            broken_nodes = []
        visited.add(node)
        logging.debug(f'Visited: {visited}')
        logging.debug(f'Node: {node}')


        if bool(graph.perturbation_resistant[node]):
            logging.debug(f'Node {node} visited, fault resistant node')
            return visited, broken_nodes

        else:
            fathers = {'AND': set(), 'OR': set(), 'SINGLE': set() }
            predecessors = list(graph.predecessors(node))
            logging.debug(f'Predecessors: {predecessors}')

            if len(visited) == 1:
                broken_nodes.append(node)
                logging.debug(f'Broken: {broken_nodes}')

            elif predecessors:
                for p in predecessors:
                    fathers[graph.father_condition[(p, node)]].add(p)

                if fathers['AND'] & set(broken_nodes):
                    broken_nodes.append(node)
                    logging.debug(f'Broken {node}, AND predecessor broken.')
                    logging.debug(f'Nodes broken so far: {broken_nodes}')

                #'SINGLE' treated as 'AND'
                elif fathers['SINGLE'] & set(broken_nodes):
                    broken_nodes.append(node)
                    logging.debug(f'Broken {node}, SINGLE predecessor broken.')
                    logging.debug(f'Nodes broken so far: {broken_nodes}')

                else:
                    if (fathers['OR'] & set(broken_nodes)) == set(predecessors):
                        #all my 'OR' predecessors are dead
                        broken_nodes.append(node)
                        logging.debug(f'Broken {node}, no more fathers')
                        logging.debug(f'Nodes broken so far: {broken_nodes}')
                    else:
                        logging.debug(f'Surviving fathers: {fathers}')
                        logging.debug(f'Nodes broken so far: {broken_nodes}')
                        return 0
            else:
                broken_nodes.append(node)
                logging.debug(f'Node: {node} has no more predecessors')
                logging.debug(f'Nodes broken so far: {broken_nodes}')

        for next_node in set(graph[node]) - visited:
            self.rm_nodes(next_node, graph, visited, broken_nodes)

        return visited, broken_nodes

    def update_output(self, attribute_list, prefix=str()):
        """

        Update columns output DataFrame with attributes
        in attribute_list.

        :param list attribute_list: list of attributes to be updated
            to the output DataFrame.
        :param prefix: prefix to be added to column name,
            default to empty string.
        :type prefix: str, optional
        """

        for col in attribute_list:
            self.df[prefix + col] = pd.Series(getattr(self.G, col))

    def update_status_areas(self, damaged_areas):
        """

        Update the status of the elements in the damaged areas
        after the propagation of the perturbation.

        :param list damaged_areas: area(s) in which to update the status.
        """

        self.df['mark_status'].fillna('NOT_ACTIVE', inplace=True)

        old_status_area = self.G.status_area
        new_status_area = old_status_area
        for node in old_status_area.keys():
            if self.G.area[node] in damaged_areas:
                new_status_area[node] = 'DAMAGED'

        self.G.status_area = new_status_area

        for area in damaged_areas:
            self.df.loc[self.df.area == area, 'status_area'] = 'DAMAGED'

    def delete_a_node(self, node):
        """

        Delete a node in the graph.

        :param str node: the id of the node to remove.

        .. warning:: the node id must be contained in the graph.
            No check is done within this function.
        """

        _ , broken_nodes = self.rm_nodes(node, self.G)
        broken_nodes = list(set(broken_nodes))

        for n in broken_nodes:
            self.damaged_areas.add(self.G.area[n])
            self.G.remove_node(n)

    def apply_perturbation(self, perturbed_nodes, params, kind='element'):
        """

        Perturbation simulator, actually applying the perturbation
        to all the nodes affected by the perturbation.
        The optimizer is run if any switch is present, and edges connecting
        its predecessors are removed if the switch state is set to 'False'.

        :param list perturbed_nodes: nodes(s) involved in the
            perturbing event.
        :param dict params: values for the optimizer evolutionary algorithm.
            Dict of: {str: int, str: int, str: float, str: float, str: int}.

            'npop' -- number of individuals for each population (default to 300)
            'ngen' -- total number of generations (default to 100)
            'indpb' -- independent probability for attributes to be changed
                (default to 0.6)
            'tresh' -- threshold for applying crossover/mutation
                (default to 0.5)
            'nsel' -- number of individuals to select (default to 5)
        :param str kind: type of simulation, used to label output files,
            default to 'element'

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas.
        """

        self.check_before()

        self.G.clear_data(['shortest_path', 'shortest_path_length',
            'efficiency', 'nodal_efficiency', 'local_efficiency',
            'computed_service', 'closeness_centrality',
            'betweenness_centrality', 'indegree_centrality',
            'outdegree_centrality', 'degree_centrality'])

        if self.G.switches:
            res = np.array(self.optimizer(perturbed_nodes, self.G.init_status,
                params))
            best = dict(zip(self.G.init_status.keys(),
                res[np.argmin(res[:, 1]), 0]))

            for switch, opened in best.items():
                if not opened:
                    for pred in list(self.G.predecessors(switch)):
                        self.G.remove_edge(pred, switch)

            logging.debug(f'BEST: {best}, with fitness: {np.min(res[:, 1])}')
            self.G.final_status = best
 
        for node in perturbed_nodes:
            if node in self.G.nodes(): self.delete_a_node(node)

        self.check_after()
        self.paths_df.to_csv('service_paths_' + str(kind)+ '_perturbation.csv',
            index=False)

        status_area_fields = ['final_status', 'mark_status', 'status_area']
        self.update_output(status_area_fields)

        self.update_status_areas(self.damaged_areas)
        self.graph_characterization_to_file(str(kind) + '_perturbation.csv')

    def simulate_element_perturbation(self, perturbed_nodes,
        params={'npop': 300, 'ngen': 100, 'indpb': 0.6, 'tresh': 0.5,
        'nsel': 5}):
        """

        Simulate a perturbation of one or multiple nodes.

        :param list perturbed_nodes: nodes(s) involved in the
            perturbing event.
        :param dict params: values for the optimizer evolutionary algorithm.
            Dict of: {str: int, str: int, str: float, str: float, str: int}.

            'npop' -- number of individuals for each population (default to 300)
            'ngen' -- total number of generations (default to 100)
            'indpb' -- independent probability for attributes to be changed
                (default to 0.6)
            'tresh' -- threshold for applying crossover/mutation
                (default to 0.5)
            'nsel' -- number of individuals to select (default to 5)

        :raises: SystemExit
        """

        for node in perturbed_nodes:

            if node not in self.G.nodes():
                logging.debug(f'The node {node} is not in the graph')
                logging.debug('Insert a valid node')
                logging.debug(f'Valid nodes: {self.G.nodes()}')
                sys.exit()

        self.apply_perturbation(perturbed_nodes, params, kind='element')

    def simulate_area_perturbation(self, perturbed_areas, params={'npop': 300,
        'ngen': 100, 'indpb': 0.6, 'tresh': 0.5, 'nsel': 5}):
        """

        Simulate a perturbation in one or multiple areas.

        :param list perturbed_areas: area(s) involved in the
            perturbing event.
        :param dict params: values for the optimizer evolutionary algorithm.
            Dict of: {str: int, str: int, str: float, str: float, str: int}.

            'npop' -- number of individuals for each population (default to 300)
            'ngen' -- total number of generations (default to 100)
            'indpb' -- independent probability for attributes to be changed
                (default to 0.6) 
            'tresh' -- threshold for applying crossover/mutation
                (default to 0.5)
            'nsel' -- number of individuals to select (default to 5)

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas.

        :raises: SystemExit
        """

        nodes_in_area = []

        for area in perturbed_areas:

            if area not in list(self.G.area.values()):
                logging.debug(f'The area {area} is not in the graph')
                logging.debug('Insert a valid area')
                logging.debug(f'Valid areas: {set(self.G.area.values())}')
                sys.exit()
            else:
                for idx, idx_area in self.G.area.items():
                    if idx_area == area: nodes_in_area.append(idx)

        self.apply_perturbation(nodes_in_area, params, kind='area')

    def graph_characterization_to_file(self, filename):
        """

        Write to file graph characterization after the perturbation.
        File is written in CSV format.

        :param str filename: output file name where to print the
            graph characterization.
        """

        self.df.reset_index(inplace=True)
        self.df.rename(columns={'index': 'mark'}, inplace=True)

        fields = [
            'mark', 'description', 'init_status', 'final_status',
            'mark_status', 'perturbation_resistant', 'area', 'status_area',
            'closeness_centrality', 'betweenness_centrality',
            'indegree_centrality', 'outdegree_centrality',
            'original_local_efficiency', 'final_local_efficiency',
            'original_nodal_efficiency', 'final_nodal_efficiency',
            'original_service', 'final_service'
        ]
        self.df[fields].to_csv(filename, index=False)
