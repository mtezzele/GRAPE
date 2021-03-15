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

        :param str filename: input file in CSV format
        """

        if parallel:
            self.G = ParallelGeneralGraph()
        else:
            self.G = GeneralGraph()

        self.df, self.edges_df = self.G.load(filename)

        self.damaged_areas = set()

        self.valv = {  "isolation_A" : { "0": "OPEN", "1": "CLOSED"},
           "isolation_B" : { "0": "CLOSED", "1": "OPEN"},
           "unknown" : { "0": "OFF", "1": "ON"} }

    def check_input_with_gephi(self):
        """

        Write list of nodes and list of edges csv files
        to visualize the input with Gephi.
        """

        gephi_nodes_df = self.df.reset_index()
        gephi_nodes_df.rename(columns={'index': 'Mark'}, inplace=True)

        fields = [ "Mark", "Description", "InitStatus",
                   "PerturbationResistant", "Area" ]

        gephi_nodes_df[fields].to_csv("check_import_nodes.csv", index=False)

        orphans = self.edges_df['Father_mark'].str.contains("NULL")
        self.edges_df = self.edges_df[~orphans]
        self.edges_df.to_csv("check_import_edges.csv", index=False)

    def check_before(self):
        """

        Describe the topology of the integer graph, before the
        occurrence of any perturbation in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.
        """

        self.G.calculate_shortest_path()
        original_source_user_paths = []

        self.G.nodal_efficiency()
        self.G.local_efficiency()
        self.G.compute_service()

        eff_fields = ['nodal_eff', 'local_eff', 'service']
        self.update_output(eff_fields, prefix="original_")

        for source in self.G.SOURCE:
            for user in self.G.USER:
                if nx.has_path(self.G, source, user):

                    osip = list(nx.all_simple_paths(self.G, source, user))
                    oshp = self.G.nodes[source]["shortest_path"][user]
                    oshpl = self.G.nodes[source]["shpath_length"][user]
                    oeff = 1 / oshpl
                    ids = source + user

                else:
                    oshpl = "NO_PATH"
                    osip = "NO_PATH"
                    oshp = "NO_PATH"
                    oeff = "NO_PATH"
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
        self.G.closeness_centrality()
        self.G.betweenness_centrality()
        self.G.indegree_centrality()
        self.G.outdegree_centrality()
        self.G.degree_centrality()

    def check_after(self):
        """

        Describe the topology of the potentially perturbed graph,
        after the occurrence of a perturbation in the system.
        Compute efficiency measures for the whole graph and its nodes.
        Check the availability of paths between source and target nodes.
        """

        self.G.calculate_shortest_path()
        final_source_user_paths = []

        self.G.nodal_efficiency()
        self.G.local_efficiency()
        self.G.compute_service()

        eff_fields = ['nodal_eff', 'local_eff', 'service']
        self.update_output(eff_fields, prefix="final_")

        for source in self.G.SOURCE:
            for user in self.G.USER:
                if nx.has_path(self.G, source, user):

                    sip = list(nx.all_simple_paths(self.G, source, user))
                    set_sip = set(x for lst in sip for x in lst)

                    for node in set_sip:

                        if self.G.description[node] in self.valv:

                            if self.G.nodes[node]['IntermediateStatus'] == "1":

                                logging.debug(
                                "valve %s at node %s, state %s",
                                self.G.description[node], node,
                                self.valv[self.G.description[node]]["1"])

                            elif self.G.nodes[node]['IntermediateStatus']== "0":

                                self.G.nodes[node]['FinalStatus'] = "1"

                                logging.debug(
                                "valve %s at node %s, from %s to %s",
                                self.G.description[node], node,
                                self.valv[self.G.description[node]]["0"],
                                self.valv[self.G.description[node]]["1"])

                            else:

                                if self.G.status[node] == "1":

                                    logging.debug(
                                    "valve %s at node %s, state %s",
                                    self.G.description[node], node,
                                    self.valv[self.G.description[node]]["1"])

                                elif self.G.status[node] == "0":

                                    self.G.nodes[node]['FinalStatus'] = "1"

                                    logging.debug(
                                    "valve %s at node %s, from %s to %s",
                                    self.G.description[node], node,
                                    self.valv[self.G.description[node]]["0"],
                                    self.valv[self.G.description[node]]["1"])

                    shp = self.G.nodes[source]["shortest_path"][user]
                    shpl = self.G.nodes[source]["shpath_length"][user]
                    neff = 1 / shpl
                    ids = source + user

                else:

                    shpl = "NO_PATH"
                    sip = "NO_PATH"
                    shp = "NO_PATH"
                    neff = "NO_PATH"
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
        They are deleted, instead, if object of node deletion
        themselves.

        :param str node: the id of the node to remove
        :param visited: list of nodes already visited
        :type visited: set, optional
        """

        if visited is None:
            visited = set()
        visited.add(node)
        logging.debug('Visited: %s', visited)
        logging.debug('Node: %s', node)

        if self.G.fault_resistant[node] == "1":
            logging.debug('Node %s visited, fault resistant node', node)
            return visited

        if self.G.description[node] in self.valv:

            if self.G.status[node] == "0":
                logging.debug('Valve %s at node %s, state %s',
                self.G.description[node], node,
                self.valv[self.G.description[node]]["0"])

            elif self.G.status[node] == "1":

                self.G.nodes[node]['IntermediateStatus'] = "0"

                logging.debug(
                'Valve %s at node %s, from %s to %s',
                self.G.description[node], node,
                self.valv[self.G.description[node]]["1"],
                self.valv[self.G.description[node]]["0"])

            if len(visited) == 1:
                self.broken.append(node)
                logging.debug("Valve perturbed: %s", self.broken)

            else:
                return visited

        else:
            fathers = {"AND": set(), "OR": set(), "SINGLE": set() }
            pred = list(self.G.predecessors(node))
            logging.debug("Predecessors: %s", pred)

            if len(visited) == 1:
                self.broken.append(node)
                logging.debug("Broken: %s", self.broken)

            elif pred:
                for p in pred:
                    fathers[self.G.condition[(p, node)]].add(p)

                if fathers["AND"] & set(self.broken):
                    self.broken.append(node)
                    logging.debug("Broken %s, AND predecessor broken.", node)
                    logging.debug("Nodes broken so far: %s", self.broken)

                #"SINGLE" treated as "AND"
                elif fathers["SINGLE"] & set(self.broken):
                    self.broken.append(node)
                    logging.debug("Broken %s, SINGLE predecessor broken.", node)
                    logging.debug("Nodes broken so far: %s", self.broken)

                else:
                    #all my "OR" predecessors are dead
                    if (fathers["OR"] & set(self.broken)) == set(pred):
                        self.broken.append(node)
                        logging.debug("Broken %s, no more fathers", node)
                        logging.debug("Nodes broken so far: %s", self.broken)
                    else:
                        return 0
            else:
                self.broken.append(node)
                logging.debug("Node: %s has no more predecessors", node)
                logging.debug("Nodes broken so far: %s", self.broken)

        for next_node in set(self.G[node]) - visited:
            self.rm_nodes(next_node, visited)

        return visited

    def update_output(self, attribute_list, prefix=str()):
        """

        Update columns output DataFrame with attributes
        in attribute_list.

        :param list attribute_list: list of attributes to be updated
            to the DataFrame
        :param prefix: prefix to be added to column name
        :type prefix: str, optional
        """

        nested_dict = {}
        for col in attribute_list:
            nested_dict[prefix + col] = nx.get_node_attributes(self.G, col)

        self.df = pd.concat([self.df, pd.DataFrame(nested_dict)], axis=1)

    def update_status_areas(self, damaged_areas):
        """

        Update the status of the elements in the damaged areas
        after the propagation of the perturbation.

        :param list damaged_areas: area(s) in which to update the status
        """

        self.df['Mark_Status'].fillna('NOT_ACTIVE', inplace=True)

        for area in damaged_areas:
            self.df.loc[self.df.Area == area, 'Status_Area'] = "DAMAGED"

    def delete_a_node(self, node):
        """

        Delete a node in the graph.

        :param str node: the id of the node to remove

        .. note:: the node id must be contained in the graph.
            No check is done within this function.
        """

        self.broken = [] #clear previous perturbation broken nodes

        self.rm_nodes(node)
        self.bn = list(set(self.broken))

        for n in self.bn:
            self.damaged_areas.add(self.G.nodes[n]["Area"])
            self.G.remove_node(n)

    def simulate_element_perturbation(self, perturbed_nodes):
        """

        Simulate a perturbation of one or multiple nodes.
        Nodes' "IntermediateStatus", "FinalStatus", "Mark_Status"
        and "Status_Area" attributes are evaluated.

        :param list perturbed_nodes: nodes(s) involved in the
            perturbing event

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas.
        """

        for node in perturbed_nodes:

            if node not in self.G.nodes():
                print('The node ', node, ' is not in the graph')
                print('Insert a valid node')
                print("Valid nodes:", self.G.nodes())
                sys.exit()

        self.check_before()

        centrality_fields = ['closeness_centrality', 'betweenness_centrality',
            'indegree_centrality', 'outdegree_centrality', 'degree_centrality']
        self.update_output(centrality_fields)

        for node in perturbed_nodes:
            if node in self.G.nodes():
                self.delete_a_node(node)

        del_sources = [s for s in self.G.SOURCE if s not in list(self.G)]
        for s in del_sources:
            self.G.SOURCE.remove(s)

        del_users = [u for u in self.G.USER if u not in list(self.G)]
        for u in del_users:
            self.G.USER.remove(u)

        self.check_after()
        self.paths_df.to_csv("service_paths_element_perturbation.csv",
            index=False)

        status_area_fields = ['IntermediateStatus', 'FinalStatus',
            'Mark_Status', 'Status_Area']
        self.update_output(status_area_fields)

        self.update_status_areas(self.damaged_areas)
        self.graph_characterization_to_file("element_perturbation.csv")

    def simulate_area_perturbation(self, perturbed_areas):
        """

        Simulate a perturbation in one or multiple areas.
        Nodes' "IntermediateStatus", "FinalStatus", "Mark_Status"
        and "Status_Area" attributes are evaluated.

        :param list perturbed_areas: area(s) involved in the
            perturbing event

        .. note:: A perturbation, depending on the considered system,
            may spread in all directions starting from the damaged
            component(s) and may be affect nearby areas
        """

        nodes_in_area = []

        for area in perturbed_areas:

            if area not in list(self.G.area.values()):
                print('The area ', area, ' is not in the graph')
                print('Insert a valid area')
                print("Valid areas:", set(self.G.area.values()))
                sys.exit()
            else:
                for idx, Area in self.G.area.items():
                    if Area == area:
                        nodes_in_area.append(idx)

        self.check_before()

        centrality_fields = ['closeness_centrality', 'betweenness_centrality',
            'indegree_centrality', 'outdegree_centrality', 'degree_centrality']
        self.update_output(centrality_fields)

        for node in nodes_in_area:
            if node in self.G.nodes():
                self.delete_a_node(node)
                nodes_in_area = list(set(nodes_in_area) - set(self.bn))

        del_sources = [s for s in self.G.SOURCE if s not in list(self.G)]
        for s in del_sources:
            self.G.SOURCE.remove(s)

        del_users = [u for u in self.G.USER if u not in list(self.G)]
        for u in del_users:
            self.G.USER.remove(u)

        self.check_after()
        self.paths_df.to_csv("service_paths_area_perturbation.csv", index=False)

        status_area_fields = ['IntermediateStatus', 'FinalStatus',
            'Mark_Status', 'Status_Area']
        self.update_output(status_area_fields)

        self.update_status_areas(self.damaged_areas)
        self.graph_characterization_to_file("area_perturbation.csv")

    def graph_characterization_to_file(self, filename):
        """

        Write to file graph characterization
        after the perturbation.

        :param str filename: output file name where to print the
            graph characterization
        """

        self.df.reset_index(inplace=True)
        self.df.rename(columns={'index': 'Mark'}, inplace=True)

        fields = [
            "Mark", "Description", "InitStatus", "IntermediateStatus",
            "FinalStatus", "Mark_Status", "PerturbationResistant", "Area",
            "Status_Area", "closeness_centrality", "betweenness_centrality",
            "indegree_centrality", "outdegree_centrality",
            "original_local_eff", "final_local_eff",
            "original_nodal_eff", "final_nodal_eff",
            "original_service", "final_service"
        ]
        self.df[fields].to_csv(filename, index=False)
