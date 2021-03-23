"""FaultDiagnosis module"""

import logging
import sys
import warnings
import networkx as nx
import pandas as pd

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

        self.valv = {  'isolation_A' : { '0': 'OPEN', '1': 'CLOSED'},
           'isolation_B' : { '0': 'CLOSED', '1': 'OPEN'},
           'unknown' : { '0': 'OFF', '1': 'ON'} }

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

        for source in self.G.source:
            for user in self.G.user:
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

        for source in self.G.source:
            for user in self.G.user:
                if nx.has_path(self.G, source, user):

                    sip = list(nx.all_simple_paths(self.G, source, user))
                    set_sip = set(x for lst in sip for x in lst)

                    for node in set_sip:

                        if self.G.description[node] in self.valv:

                            if self.G.intermediate_status[node] == '1':

                                valve = self.G.description[node]
                                state = self.valv[self.G.description[node]]['1']
                                logging.debug(
                                f'Valve {valve} at node {node}, state {state}')

                            elif self.G.intermediate_status[node]== '0':

                                self.G.nodes[node]['final_status'] = '1'

                                valve = self.G.description[node]
                                old = self.valv[self.G.description[node]]['0']
                                new = self.valv[self.G.description[node]]['1']
                                logging.debug(f'Valve {valve} at node {node},')
                                logging.debug(f'from {old} to {new}')

                            else:

                                if self.G.init_status[node] == '1':

                                    valve = self.G.description[node]
                                    s = self.valv[self.G.description[node]]['1']
                                    logging.debug(
                                    f'Valve {valve} at node {node}, state {s}')

                                elif self.G.init_status[node] == '0':

                                    self.G.nodes[node]['final_status'] = '1'

                                    valve = self.G.description[node]
                                    o = self.valv[self.G.description[node]]['0']
                                    n = self.valv[self.G.description[node]]['1']
                                    logging.debug(
                                    f'Valve {valve} at node {node}')
                                    logging.debug(f'from {o} to {n}')

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

    def rm_nodes(self, node, visited=None):
        """

        Remove nodes from the graph in a depth first search way to
        propagate the perturbation.
        Nodes are not deleted if perturbation resistant.
        Moreover, valves are not deleted if encountered
        during the propagation of a the perturbation.
        They are deleted, instead, if object of node deletion themselves.

        :param str node: the id of the node to remove.
        :param visited: list of nodes already visited,
            default to None.
        :type visited: set, optional
        """

        if visited is None:
            visited = set()
        visited.add(node)
        logging.debug(f'Visited: {visited}')
        logging.debug(f'Node: {node}')

        if self.G.perturbation_resistant[node] == "1":
            logging.debug(f'Node {node} visited, fault resistant node')
            return visited

        if self.G.description[node] in self.valv:

            if self.G.init_status[node] == '0':

                valve = self.G.description[node]
                state = self.valv[self.G.description[node]]['0'] 
                logging.debug(f'Valve {valve} at node {node}, state {state}')

            elif self.G.init_status[node] == '1':

                self.G.nodes[node]['intermediate_status'] = '0'

                valve = self.G.description[node]
                old = self.valv[self.G.description[node]]['1']
                new = self.valv[self.G.description[node]]['0']

                logging.debug(
                f'Valve {valve} at node {node}), from {old} to {new}')

            if len(visited) == 1:
                self.broken.append(node)
                logging.debug(f'Valve perturbed: {self.broken}')

            else:
                return visited

        else:
            fathers = {'AND': set(), 'OR': set(), 'SINGLE': set() }
            predecessors = list(self.G.predecessors(node))
            logging.debug(f'Predecessors: {predecessors}')

            if len(visited) == 1:
                self.broken.append(node)
                logging.debug(f'Broken: {self.broken}')

            elif predecessors:
                for p in predecessors:
                    fathers[self.G.father_condition[(p, node)]].add(p)

                if fathers['AND'] & set(self.broken):
                    self.broken.append(node)
                    logging.debug(f'Broken {node}, AND predecessor broken.')
                    logging.debug(f'Nodes broken so far: {self.broken}')

                #'SINGLE' treated as 'AND'
                elif fathers['SINGLE'] & set(self.broken):
                    self.broken.append(node)
                    logging.debug(f'Broken {node}, SINGLE predecessor broken.')
                    logging.debug(f'Nodes broken so far: {self.broken}')

                else:
                    #all my 'OR' predecessors are dead
                    if (fathers['OR'] & set(self.broken)) == set(pred):
                        self.broken.append(node)
                        logging.debug(f'Broken {node}, no more fathers')
                        logging.debug(f'Nodes broken so far: {self.broken}')
                    else:
                        return 0
            else:
                self.broken.append(node)
                logging.debug(f'Node: {node} has no more predecessors')
                logging.debug(f'Nodes broken so far: {self.broken}')

        for next_node in set(self.G[node]) - visited:
            self.rm_nodes(next_node, visited)

        return visited

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

        for area in damaged_areas:
            self.df.loc[self.df.area == area, 'status_area'] = 'DAMAGED'

    def delete_a_node(self, node):
        """

        Delete a node in the graph.

        :param str node: the id of the node to remove.

        .. warning:: the node id must be contained in the graph.
            No check is done within this function.
        """

        self.broken = [] #clear previous perturbation broken nodes

        self.rm_nodes(node)
        self.bn = list(set(self.broken))

        for n in self.bn:
            self.damaged_areas.add(self.G.area[n])
            self.G.remove_node(n)

    def simulate_element_perturbation(self, perturbed_nodes):
        """

        Simulate a perturbation of one or multiple nodes.

        :param list perturbed_nodes: nodes(s) involved in the
            perturbing event.

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas.

        :raises: SystemExit
        """

        for node in perturbed_nodes:

            if node not in self.G.nodes():
                logging.debug(f'The node {node} is not in the graph')
                logging.debug('Insert a valid node')
                logging.debug(f'Valid nodes: {self.G.nodes()}')
                sys.exit()

        self.check_before()

        self.G.clear_data(['shortest_path', 'shortest_path_length',
            'efficiency', 'nodal_efficiency', 'local_efficiency',
            'computed_service', 'closeness_centrality',
            'betweenness_centrality', 'indegree_centrality',
            'outdegree_centrality', 'degree_centrality'])

        for node in perturbed_nodes:
            if node in self.G.nodes(): self.delete_a_node(node)

        deleted_sources = [s for s in self.G.source if s not in list(self.G)]
        for s in deleted_sources: self.G.source.remove(s)

        deleted_users = [u for u in self.G.user if u not in list(self.G)]
        for u in deleted_users: self.G.user.remove(u)

        self.check_after()
        self.paths_df.to_csv('service_paths_element_perturbation.csv',
            index=False)

        status_area_fields = ['intermediate_status', 'final_status',
            'mark_status', 'status_area']
        self.update_output(status_area_fields)

        self.update_status_areas(self.damaged_areas)
        self.graph_characterization_to_file('element_perturbation.csv')

    def simulate_area_perturbation(self, perturbed_areas):
        """

        Simulate a perturbation in one or multiple areas.

        :param list perturbed_areas: area(s) involved in the
            perturbing event.

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas.

        :raises: SystemExit
        """

        nodes_in_area = []

        for area in perturbed_areas:

            if area not in list(self.G.area.values()):
                logging.debug(f'The area {area} is not in the graph')
                print('Insert a valid area')
                print(f'Valid areas: {set(self.G.area.values())}')
                sys.exit()
            else:
                for idx, idx_area in self.G.area.items():
                    if idx_area == area: nodes_in_area.append(idx)

        self.check_before()

        self.G.clear_data(['shortest_path', 'shortest_path_length',
            'efficiency', 'nodal_efficiency', 'local_efficiency',
            'computed_service', 'closeness_centrality',
            'betweenness_centrality', 'indegree_centrality',
            'outdegree_centrality', 'degree_centrality'])

        for node in nodes_in_area:
            if node in self.G.nodes():
                self.delete_a_node(node)
                nodes_in_area = list(set(nodes_in_area) - set(self.bn))

        deleted_sources = [s for s in self.G.source if s not in list(self.G)]
        for s in deleted_sources: self.G.source.remove(s)

        deleted_users = [u for u in self.G.user if u not in list(self.G)]
        for u in deleted_users: self.G.user.remove(u)

        self.check_after()
        self.paths_df.to_csv('service_paths_area_perturbation.csv', index=False)

        status_area_fields = ['intermediate_status', 'final_status',
            'mark_status', 'status_area']
        self.update_output(status_area_fields)

        self.update_status_areas(self.damaged_areas)
        self.graph_characterization_to_file('area_perturbation.csv')

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
            'mark', 'description', 'init_status', 'intermediate_status',
            'final_status', 'mark_status', 'perturbation_resistant', 'area',
            'status_area', 'closeness_centrality', 'betweenness_centrality',
            'indegree_centrality', 'outdegree_centrality',
            'original_local_efficiency', 'final_local_efficiency',
            'original_nodal_efficiency', 'final_nodal_efficiency',
            'original_service', 'final_service'
        ]
        self.df[fields].to_csv(filename, index=False)
