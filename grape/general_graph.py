"""GeneralGraph for directed graphs (DiGraph) module"""

import logging
import sys
import warnings
import copy
import numpy as np
import networkx as nx
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class GeneralGraph(nx.DiGraph):
    """Class GeneralGraph for directed graphs (DiGraph).

    Constructs a new graph given an input file.
    A DiGraph stores nodes and edges with optional data or attributes.
    DiGraphs hold directed edges.
    Nodes can be arbitrary python objects with optional key/value attributes.
    Edges are represented  as links between nodes with optional key/value
    attributes.
    """

    def __init__(self):
        super().__init__()

    def load(self, filename):
        """

        Load input file. Input file must be in CSV format.
        Each line corresponds to a node/element description,
        with the relative hierarchy, together with the list
        of all the node attributes.

        :param str filename: input file in CSV format
        """

        conv = {'Mark' : str, 'Father_mark' : str,
                'PerturbationResistant':str, 'InitStatus':str}
        graph_df = pd.read_csv(filename, converters=conv, keep_default_na=False)

        for index, row in graph_df.iterrows():

            weight = row.pop('Weight')
            father_mark = row.pop('Father_mark')
            father_cond = row.pop('Father_cond')

            self.add_node(row['Mark'], **row)

            if father_mark != 'NULL':
                self.add_edge(
                    father_mark, row['Mark'],
                    Father_cond=father_cond, weight=weight)

        graph_edges_df = graph_df[['Mark', 'Father_mark']]

        graph_df.drop(['Father_cond', 'Father_mark', 'Type',
                      'Weight', 'Service'], axis=1, inplace=True)
        graph_df.drop_duplicates(inplace=True)
        graph_df.set_index('Mark', inplace=True)

        nx.set_node_attributes(self, str(), 'IntermediateStatus')
        nx.set_node_attributes(self, str(), 'FinalStatus')
        nx.set_node_attributes(self, 'AVAILABLE', 'Status_Area')
        nx.set_node_attributes(self, 'ACTIVE', 'Mark_Status')

        self.SOURCE = []
        self.USER = []
        for idx, Type in self.type.items():
            if Type == 'SOURCE':
                self.SOURCE.append(idx)
            elif Type == 'USER':
                self.USER.append(idx)

        return graph_df, graph_edges_df

    @property
    def area(self):
        return nx.get_node_attributes(self, 'Area')

    @property
    def fault_resistant(self):
        return nx.get_node_attributes(self, 'PerturbationResistant')

    @property
    def description(self):
        return nx.get_node_attributes(self, 'Description')

    @property
    def init_status(self):
        return nx.get_node_attributes(self, 'InitStatus')

    @property
    def intermediate_status(self):
        return nx.get_node_attributes(self, 'IntermediateStatus')

    @property
    def final_status(self):
        return nx.get_node_attributes(self, 'FinalStatus')

    @property
    def mark_status(self):
        return nx.get_node_attributes(self, 'Mark_Status')

    @property
    def status_area(self):
        return nx.get_node_attributes(self, 'Status_Area')

    @property
    def condition(self):
        return nx.get_edge_attributes(self, 'Father_cond')

    @property
    def type(self):
        return nx.get_node_attributes(self, 'Type')

    @property
    def initial_service(self):
        return nx.get_node_attributes(self, 'Service')

    @property
    def service(self):

        computed_service = nx.get_node_attributes(self, 'computed_service')
        if computed_service: return computed_service

        computed_service, _ = self.compute_service()
        nx.set_node_attributes(self, computed_service, name='computed_service')
        return computed_service

    @property
    def shortest_path(self):
        """

        Shortest existing paths between all node pairs.
        """

        shortest_path = nx.get_node_attributes(self, 'shortest_path')
        if shortest_path: return shortest_path

        shortest_path, shortest_path_length = self.calculate_shortest_path()
        nx.set_node_attributes(self, shortest_path, name='shortest_path')
        nx.set_node_attributes(self, shortest_path_length,
            name='shortest_path_length')
        return shortest_path

    @property
    def shortest_path_length(self):
        """

        Shortest path length.
        """

        shortest_path_length = nx.get_node_attributes(self,
            'shortest_path_length')
        if shortest_path_length: return shortest_path_length

        shortest_path, shortest_path_length = self.calculate_shortest_path()
        nx.set_node_attributes(self, shortest_path, name='shortest_path')
        nx.set_node_attributes(self, shortest_path_length,
            name='shortest_path_length')
        return shortest_path_length

    @property
    def efficiency(self):
        """

        Efficiency of the graph.
        Returns the efficiency if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        efficiency = nx.get_node_attributes(self, 'efficiency')
        if efficiency: return efficiency

        efficiency = self.compute_efficiency()
        nx.set_node_attributes(self, efficiency, name='efficiency')
        return efficiency

    @property
    def nodal_efficiency(self):
        """

        Nodal efficiency of the graph.
        Returns the nodal efficiency if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        nodal_eff = nx.get_node_attributes(self, 'nodal_efficiency')
        if nodal_eff: return nodal_eff

        nodal_eff = self.compute_nodal_efficiency()
        nx.set_node_attributes(self, nodal_eff, name='nodal_efficiency')
        return nodal_eff

    @property
    def local_efficiency(self):
        """

        Local efficiency of the graph.
        Returns the local efficiency if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        local_eff = nx.get_node_attributes(self, 'local_efficiency')
        if local_eff: return local_eff

        local_eff = self.compute_local_efficiency()
        nx.set_node_attributes(self, local_eff, name='local_efficiency')
        return local_eff

    @property
    def global_efficiency(self):
        """

        Average global efficiency of the whole graph.

        .. note:: The average global efficiency of a graph is the average
            efficiency of all pairs of nodes.
        """

        g_len = len(list(self))
        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        nodeff_values = list(self.nodal_efficiency.values())
        return sum(nodeff_values) / g_len

    @property
    def betweenness_centrality(self):
        """

        Betweenness centrality of the graph.
        Returns the betweenness centrality if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        bet_cen = nx.get_node_attributes(self, 'betweenness_centrality')
        if bet_cen: return bet_cen

        bet_cen = self.compute_betweenness_centrality()
        nx.set_node_attributes(self, bet_cen, name='betweenness_centrality')
        return bet_cen

    @property
    def closeness_centrality(self):
        """

        Closeness centrality of the graph.
        Returns the closeness centrality if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        clo_cen = nx.get_node_attributes(self, 'closeness_centrality')
        if clo_cen: return clo_cen

        clo_cen = self.compute_closeness_centrality()
        nx.set_node_attributes(self, clo_cen, name='closeness_centrality')
        return clo_cen

    @property
    def degree_centrality(self):
        """

        Degree centrality of the graph.
        Returns the degree centrality if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        deg_cen = nx.get_node_attributes(self, 'degree_centrality')
        if deg_cen: return deg_cen

        deg_cen = self.compute_degree_centrality()
        nx.set_node_attributes(self, deg_cen, name='degree_centrality')
        return deg_cen

    @property
    def indegree_centrality(self):
        """

        In-degree centrality of the graph.
        Returns the in-degree centrality if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        indeg_cen = nx.get_node_attributes(self, 'indegree_centrality')
        if indeg_cen: return indeg_cen

        indeg_cen = self.compute_indegree_centrality()
        nx.set_node_attributes(self, indeg_cen, name='indegree_centrality')
        return indeg_cen

    @property
    def outdegree_centrality(self):
        """

        Out-degree centrality of the graph.
        Returns the out-degree centrality if already stored in the nodes.
        Otherwise, the attribute is computed.
        """

        outdeg_cen = nx.get_node_attributes(self, 'outdegree_centrality')
        if outdeg_cen: return outdeg_cen

        outdeg_cen = self.compute_outdegree_centrality()
        nx.set_node_attributes(self, outdeg_cen, name='outdegree_centrality')
        return outdeg_cen

    def clear_data(self, attributes_to_remove):
        """

        Delete attributes for all nodes in the graph.

        :param list attributes_to_remove: a list of strings
            with all the attributes we want to remove
        """

        for attr in attributes_to_remove:
            for node in self:
                del self.nodes[node][attr]

    def construct_path_kernel(self, nodes, pred):
        """

        Reconstruct source-target paths starting from predecessors
        matrix, and populate the dictionary of shortest paths.

        :param list nodes: list of nodes for which to compute the
            shortest path between them and all the other nodes
        :param numpy.ndarray pred: matrix of predecessors,
            computed with Floyd Warshall APSP algorithm

        :return: nested dictionary with key corresponding to
            source, while as value a dictionary keyed by target and valued
            by the source-target shortest path
        :rtype: dict
        """

        shortest_paths = {}

        for i in nodes:
            
            all_targets_paths = {}

            for j in sorted(list(self.H)):

                source = i
                target = j

                if source == target:
                    path = [source]
                else:
                    pred.astype(int)
                    curr = pred[source, target]
                    if curr != np.inf:
                        curr = int(curr)
                        path = [int(target), int(curr)]
                        while curr != source:
                            curr = int(pred[int(source), int(curr)])
                            path.append(curr)
                    else:
                        path = []

                path = list(map(self.ids.get, path))
                path = list(reversed(path))

                all_targets_paths[self.ids[j]] = path

            shortest_paths[self.ids[i]] = all_targets_paths

        return shortest_paths

    def floyd_warshall_initialization(self):
        """

        Initialization of Floyd Warshall APSP algorithm.
        The distancy matrix is mutuated by NetworkX graph adjacency
        matrix, while the predecessors matrix is initialized
        with node fathers.
        The conversion between the labels (ids) in the graph and Numpy
        matrix indices (and viceversa) is also exploited.

        .. note:: In order for the ids relation to be bijective,
            'Mark' attribute must be unique for each node.
        """

        self.H = nx.convert_node_labels_to_integers(
            self, first_label=0, label_attribute='Mark_ids')
        self.ids = nx.get_node_attributes(self.H, 'Mark_ids')
        self.ids_reversed = { value: key for key, value in self.ids.items() }

        dist = nx.to_numpy_matrix(self.H, nodelist=sorted(list(self.H)),
            nonedge=np.inf)
        np.fill_diagonal(dist, 0.)

        pred = np.full((len(self.H), len(self.H)), np.inf)
        for u, v in self.H.edges():
            pred[u, v] = u

        return dist, pred

    def floyd_warshall_kernel(self, dist, pred, init, stop, barrier=None):
        """

        Floyd Warshall's APSP inner iteration.
        Distance matrix is intended to take edges weight into account.

        :param dist: matrix of distances
        :type dist: numpy.ndarray or multiprocessing.sharedctypes.RawArray
        :param pred: matrix of predecessors
        :type pred: numpy.ndarray or multiprocessing.sharedctypes.RawArray
        :param int init: starting column of numpy matrix slice
        :param int stop: ending column of numpy matrix slice
        :param multiprocessing.synchronize.Barrier barrier:
            multiprocessing barrier to moderate writing on
            distance and predecessors matrices
        """

        n = dist.shape[0]
        for w in range(n):  # k
            dist_copy = copy.deepcopy(dist[init:stop, :])
            np.minimum(
                np.reshape(
                    np.add.outer(dist[init:stop, w], dist[w, :]),
                    (stop-init, n)),
                dist[init:stop, :],
                dist[init:stop, :])
            diff = np.equal(dist[init:stop, :], dist_copy)
            pred[init:stop, :][~diff] = np.tile(pred[w, :],
                (stop-init, 1))[~diff]

        if barrier:
            barrier.wait()

    def floyd_warshall_predecessor_and_distance(self):
        """

        Serial Floyd Warshall's APSP algorithm. The predecessors
        and distance matrices are evaluated, together with the nested
        dictionaries for shortest-path, length of the paths and
        efficiency attributes.

        .. note:: Edges weight is taken into account in the distance matrix.
            Edge weight attributes must be numerical. Distances are calculated
            as sums of weighted edges traversed.
        """

        dist, pred = self.floyd_warshall_initialization()

        self.floyd_warshall_kernel(dist, pred, 0, dist.shape[0])

        all_shortest_path = self.construct_path_kernel(list(self.H), pred)

        nonempty_shortest_path = {}
        for k in all_shortest_path.keys():
            nonempty_shortest_path[k] = {
                key: value
                for key, value in all_shortest_path[k].items() if value
            }

        shortest_path_length = {}
        for i in list(self.H):

            shortest_path_length[self.ids[i]] = {}

            for key, value in nonempty_shortest_path[self.ids[i]].items():
                length_path = dist[self.ids_reversed[value[0]],
                                   self.ids_reversed[value[-1]]]
                shortest_path_length[self.ids[i]][key] =  length_path

        return nonempty_shortest_path, shortest_path_length 

    def dijkstra_single_source_shortest_path(self):
        """

        Serial SSSP algorithm based on Dijkstra’s method.
        The nested dictionaries for shortest-path, length of the paths and
        efficiency attributes are evaluated.

        .. note:: Edges weight is taken into account.
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.
        """

        shortest_path = {}
        shortest_path_length = {}
        for n in self:
            sssps = (n, nx.single_source_dijkstra(self, n, weight='weight'))
            shortest_path[n] = sssps[1][1]
            shortest_path_length[n] = sssps[1][0]

        return shortest_path, shortest_path_length

    def calculate_shortest_path(self):
        """

        Choose the most appropriate way to compute the all-pairs shortest
        path depending on graph size and density.
        For a dense graph choose Floyd Warshall algorithm.
        For a sparse graph choose SSSP algorithm based on Dijkstra's method.

        .. note:: Edge weights of the graph are taken into account
            in the computation.
        """

        n_of_nodes = self.order()
        g_density = nx.density(self)

        logging.debug(f'In the graph are present {n_of_nodes} nodes')
        if g_density <= 0.000001:
            logging.debug(f'The graph is sparse, density = {g_density}')
            shpath, shpath_len = self.dijkstra_single_source_shortest_path()
        else:
            logging.debug(f'The graph is dense, density = {g_density}')
            shpath, shpath_len = self.floyd_warshall_predecessor_and_distance()

        return shpath, shpath_len

    def efficiency_kernel(self, nodes, shortest_path_length):
        """

        Compute efficiency, starting from path length attribute.
        Efficiency is a measure of how good is the exchange of commodities
        flowing from one node to the others.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param dict shortest_path_length: nested dictionary with key
            corresponding to source, while as value a dictionary keyed by target
            and valued by the source-target efficiency

        :return: nested dictionary with key corresponding to
            source, while as value a dictionary keyed by target and valued
            by the source-target efficiency
        :rtype: dict
        """

        dict_efficiency = {}

        for n in nodes:
            dict_efficiency[n] = {}
            for key, length_path in shortest_path_length[n].items():
                if length_path != 0:
                    efficiency = 1 / length_path
                    dict_efficiency[n].update({key: efficiency})
                else:
                    efficiency = 0
                    dict_efficiency[n].update({key: efficiency})

        return dict_efficiency

    def compute_efficiency(self):
        """

        Efficiency calculation.

        .. note:: The efficiency of a path connecting two nodes is defined
            as the inverse of the path length, if the path has length non-zero,
            and zero otherwise.
        """

        shortest_path_length = self.shortest_path_length
        efficiency = self.efficiency_kernel(list(self), shortest_path_length)
        return efficiency

    def nodal_efficiency_kernel(self, nodes, efficiency, g_len):
        """

        Compute nodal efficiency, starting from efficiency attribute.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param dict efficiency: nested dictionary with key corresponding to
            source, while as value a dictionary keyed by target and valued
            by the source-target efficiency
        :param int g_len: graph size

        :return: nodal efficiency dictionary keyed by node
        :rtype: dict
        """

        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        dict_nod_eff = {}
        #efficiency = self.efficiency

        for n in nodes:
            sum_efficiencies = sum(efficiency[n].values())
            dict_nod_eff[n] = sum_efficiencies / (g_len - 1)

        return dict_nod_eff

    def compute_nodal_efficiency(self):
        """

        Nodal efficiency calculation.

        .. note:: The nodal efficiency of the node is equal to zero
            for a node without any outgoing path and equal to one if from it
            we can reach each node of the digraph.
        """

        g_len = len(list(self))
        efficiency = self.efficiency
        nod_eff = self.nodal_efficiency_kernel(list(self), efficiency, g_len)
        return nod_eff

    def local_efficiency_kernel(self, nodes, nodal_efficiency):
        """

        Compute local efficiency, starting from nodal efficiency attribute.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes

        :return: local efficiency dictionary keyed by node
        :rtype: dict
        """

        dict_loc_eff = {}
        #nodal_efficiency = self.nodal_efficiency

        for n in nodes:
            subgraph = list(self.successors(n))
            denom_subg = len(list(subgraph))

            if denom_subg != 0:
                sum_efficiencies = 0
                for w in list(subgraph):
                    kv_efficiency = nodal_efficiency[w]
                    sum_efficiencies = sum_efficiencies + kv_efficiency

                dict_loc_eff[n] = sum_efficiencies / denom_subg

            else:
                dict_loc_eff[n] = 0.

        return dict_loc_eff

    def compute_local_efficiency(self):
        """

        Local efficiency calculation.

        .. note:: The local efficiency shows the efficiency of the connections
            between the first-order outgoing neighbors of node v
            when v is removed. Equivalently, local efficiency measures
            the resilience of the digraph to the perturbation of node removal,
            i.e. if we remove a node, how efficiently its first-order outgoing
            neighbors can communicate.
            It is in the range [0, 1].
        """

        nodal_efficiency = self.nodal_efficiency
        loc_eff = self.local_efficiency_kernel(list(self), nodal_efficiency)
        return loc_eff

    def shortest_path_list_kernel(self, nodes, shortest_path):
        """

        Collect the shortest paths that contain at least two nodes.

        :param list nodes: list of nodes for which to compute the
            list of shortest paths

        :return: list of shortest paths
        :rtype: list
        """

        tot_shortest_paths_list = list()
        #shortest_path = self.shortest_path

        for n in nodes:
            node_tot_shortest_paths = shortest_path[n]
            for value in node_tot_shortest_paths.values():
                if len(value) > 1:
                    tot_shortest_paths_list.append(value)

        return tot_shortest_paths_list

    def betweenness_centrality_kernel(self, nodes, tot_shortest_paths_list):
        """

        Compute betweenness centrality, from shortest path list. 

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param tot_shortest_paths_list: list of shortest paths
            with at least two nodes
        :type tot_shortest_path_list: list or multiprocessing.managers.list

        :return: between centrality dictionary keyed by node
        :rtype: dict
        """

        dict_bet_cen = {}

        for n in nodes:
            sp_with_node = []
            for l in tot_shortest_paths_list:
                if n in l and n != l[0] and n != l[-1]:
                    sp_with_node.append(l)

            numb_sp_with_node = len(sp_with_node)
            dict_bet_cen[n] = numb_sp_with_node / len(tot_shortest_paths_list)

        return dict_bet_cen

    def compute_betweenness_centrality(self):
        """

        Betweenness_centrality calculation

        .. note:: Betweenness centrality is an index of the relative importance
            of a node and it is defined by the number of shortest paths that run
            through it.
            Nodes with the highest betweenness centrality hold the higher level
            of control on the information flowing between different nodes in
            the network, because more information will pass through them.
        """

        shortest_path = self.shortest_path
        tot_shortest_paths_list = self.shortest_path_list_kernel(list(self),
            shortest_path)

        bet_cen = self.betweenness_centrality_kernel(list(self),
            tot_shortest_paths_list)
        return bet_cen

    def closeness_centrality_kernel(self, nodes, shpath_len, tot_shpaths_list,
        g_len):
        """

        Compute betweenness centrality, from shortest path list. 

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param tot_shpaths_list: list of shortest paths
            with at least two nodes
        :type tot_shortest_path_list: list or multiprocessing.managers.list
        :param int g_len: graph size

        :return: closeness centrality dictionary keyed by node
        :rtype: dict
        """

        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        dict_clo_cen = {}
        #shortest_path_length = self.shortest_path_length

        for n in nodes:
            totsp = []
            sp_with_node = []
            for l in tot_shpaths_list:
                if n in l and n == l[-1]:
                    sp_with_node.append(l)
                    length_path = shpath_len[l[0]][l[-1]]
                    totsp.append(length_path)
            norm = len(totsp) / (g_len - 1)

            if (sum(totsp)) != 0:
                dict_clo_cen[n] = (len(totsp) / sum(totsp)) * norm
            else:
                dict_clo_cen[n] = 0

        return dict_clo_cen

    def compute_closeness_centrality(self):
        """

        Closeness centrality calculation

        .. note:: Closeness centrality measures the reciprocal of the
            average shortest path distance from a node to all other reachable
            nodes in the graph. Thus, the more central a node is, the closer
            it is to all other nodes. This measure allows to identify good
            broadcasters, that is key elements in a graph, depicting how
            closely the nodes are connected with each other.
        """

        g_len = len(list(self))
        shortest_path = self.shortest_path
        shortest_path_length = self.shortest_path_length
        tot_shortest_paths_list = self.shortest_path_list_kernel(list(self),
            shortest_path)

        clo_cen = self.closeness_centrality_kernel(list(self),
            shortest_path_length, tot_shortest_paths_list, g_len)
        return clo_cen

    def degree_centrality_kernel(self, nodes, g_len):
        """

        Compute degree centrality.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param int g_len: graph size

        :return: degree centrality dictionary keyed by node
        :rtype: dict
        """

        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        dict_deg_cen = {}

        for n in nodes:
            num_neighbor_nodes = self.degree(n, weight='weight')
            dict_deg_cen[n] = num_neighbor_nodes / (g_len - 1)

        return dict_deg_cen

    def compute_degree_centrality(self):
        """

        Degree centrality measure of each node calculation.

        .. note:: Degree centrality is a simple centrality measure that counts
            how many neighbors a node has in an undirected graph.
            The more neighbors the node has the most important it is,
            occupying a strategic position that serves as a source or conduit
            for large volumes of flux transactions with other nodes.
            A node with high degree centrality is a node with many dependencies.
        """

        g_len = len(list(self))
        deg_cen = self.degree_centrality_kernel(list(self), g_len)
        return deg_cen

    def indegree_centrality_kernel(self, nodes, g_len):
        """

        Compute in-degree centrality.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param int g_len: graph size

        :return: in-degree centrality dictionary keyed by node
        :rtype: dict
        """

        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        dict_indeg_cen = {}

        for n in nodes:
            num_incoming_nodes = self.in_degree(n, weight='weight')
            dict_indeg_cen[n] = num_incoming_nodes / (g_len - 1)

        return dict_indeg_cen

    def compute_indegree_centrality(self):
        """

        In-degree centrality calculation.

        .. note:: In-degree centrality is measured by the number of edges
            ending at the node in a directed graph. Nodes with high in-degree
            centrality are called cascade resulting nodes.
        """

        g_len = len(list(self))
        indeg_cen = self.indegree_centrality_kernel(list(self), g_len)
        return indeg_cen

    def outdegree_centrality_kernel(self, nodes, g_len):
        """

        Compute out-degree centrality.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param int g_len: graph size

        :return: out-degree dictionary keyed by node
        :rtype: dict
        """

        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        dict_outdeg_cen = {}

        for n in nodes:
            num_outcoming_nodes = self.out_degree(n, weight='weight')
            dict_outdeg_cen[n] = num_outcoming_nodes / (g_len - 1)

        return dict_outdeg_cen

    def compute_outdegree_centrality(self):
        """

        Outdegree centrality calculation.

        .. note:: Outdegree centrality is measured by the number of edges
            starting from a node in a directed graph. Nodes with high outdegree
            centrality are called cascade inititing nodes.
        """

        g_len = len(list(self))
        outdeg_cen = self.outdegree_centrality_kernel(list(self), g_len)
        return outdeg_cen

    def compute_service(self):
        """

        Compute service for every node,
        together with edge splitting.

        :param graph: Graph where the service is updated
        :type graph: networkx.DiGraph
        :param str servicename: service to populate
        """

        usr_per_node = {node: 0. for node in self}
        splitting = {edge: 0. for edge in self.edges()}
        computed_service = {node: 0. for node in self}
        initial_service = self.initial_service
        shortest_path = self.shortest_path

        usr_per_source = {
            s: [u for u in self.USER if nx.has_path(self, s, u)]
            for s in self.SOURCE
        }

        for s in self.SOURCE:
            for u in usr_per_source[s]:
                for node in self.shortest_path[s][u]:
                    usr_per_node[node] += 1.

        for s in self.SOURCE:
            for u in usr_per_source[s]:
                computed_service[u] += initial_service[s]/len(usr_per_source[s])

        #Cycle just on the edges contained in source-user shortest paths
        for s in self.SOURCE:
            for u in usr_per_source[s]:
                for idx in range(len(shortest_path[s][u]) - 1):

                    head = shortest_path[s][u][idx]
                    tail = shortest_path[s][u][idx+1]

                    splitting[(head, tail)] += 1./usr_per_node[head]

        return computed_service, splitting
