"""GeneralGraph for directed graphs (DiGraph) module"""

import copy
import numpy as np
import networkx as nx
import pandas as pd


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
            if Type == "SOURCE":
                self.SOURCE.append(idx)
            elif Type == "USER":
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
    def status(self):
        return nx.get_node_attributes(self, 'InitStatus')

    @property
    def condition(self):
        return nx.get_edge_attributes(self, 'Father_cond')

    @property
    def type(self):
        return nx.get_node_attributes(self, 'Type')

    @property
    def service(self):
        return nx.get_node_attributes(self, 'Service')

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

    def efficiency_kernel(self, nodes):
        """

        Compute efficiency, starting from path length attribute.
        Efficiency is a measure of how good is the exchange of commodities
        flowing from one node to the others.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes

        :return: nested dictionary with key corresponding to
            source, while as value a dictionary keyed by target and valued
            by the source-target efficiency
        :rtype: dict
        """

        if not nx.get_node_attributes(self, "shpath_length"):
            raise ValueError("No shortest path length attribute in the graph.")

        dict_efficiency = {}

        for n in nodes:
            dict_efficiency[n] = {}
            for key, length_path in self.nodes[n]["shpath_length"].items():
                if length_path != 0:
                    efficiency = 1 / length_path
                    dict_efficiency[n].update({key: efficiency})
                else:
                    efficiency = 0
                    dict_efficiency[n].update({key: efficiency})

        return dict_efficiency

    def floyd_warshall_initialization(self):
        """

        Initialization of Floyd Warshall APSP algorithm.
        The distancy matrix is mutuated by NetworkX graph adjacency
        matrix, while the predecessors matrix is initialized
        with node fathers.
        The conversion between the labels (ids) in the graph and Numpy
        matrix indices (and viceversa) is also exploited.

        .. note:: In order for the ids relation to be bijective,
            "Mark" attribute must be unique for each node.
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
            pred[init:stop, :][~diff] = \
            np.tile(pred[w, :], (stop-init, 1))[~diff]

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

        shpaths_dicts = self.construct_path_kernel(list(self.H), pred)

        for k in shpaths_dicts.keys():
            self.nodes[k]["shortest_path"] = {
                key: value
                for key, value in shpaths_dicts[k].items() if value
            }

        for i in list(self.H):

            self.nodes[self.ids[i]]["shpath_length"] = {}

            for key, value in self.nodes[self.ids[i]]["shortest_path"].items():
                length_path = dist[self.ids_reversed[value[0]],
                                   self.ids_reversed[value[-1]]]
                self.nodes[self.ids[i]]["shpath_length"][key] =  length_path

        eff_dicts = self.efficiency_kernel(list(self))
        nx.set_node_attributes(self, eff_dicts, name="efficiency")

    def dijkstra_single_source_shortest_path(self):
        """

        Serial SSSP algorithm based on Dijkstra’s method.
        The nested dictionaries for shortest-path, length of the paths and
        efficiency attributes are evaluated.

        .. note:: Edges weight is taken into account.
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.
        """

        for n in self:
            sssps = (n, nx.single_source_dijkstra(self, n, weight = 'weight'))
            self.nodes[n]["shortest_path"] = sssps[1][1]
            self.nodes[n]["shpath_length"] = sssps[1][0]

        eff_dicts = self.efficiency_kernel(list(self))
        nx.set_node_attributes(self, eff_dicts, name="efficiency")

    def nodal_efficiency_kernel(self, nodes, g_len):
        """

        Compute nodal efficiency, starting from efficiency attribute.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes
        :param int g_len: graph size

        :return: nodal efficiency dictionary keyed by node
        :rtype: dict
        """

        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")

        if not nx.get_node_attributes(self, "efficiency"):
            raise ValueError("No efficiency attribute in the graph.")

        dict_nodeff = {}

        for n in nodes:
            sum_efficiencies = sum(self.nodes[n]["efficiency"].values())
            dict_nodeff[n] = sum_efficiencies / (g_len - 1)

        return dict_nodeff

    def nodal_efficiency(self):
        """

        Nodal efficiency.

        .. note:: The nodal efficiency of the node is equal to zero
            for a node without any outgoing path and equal to one if from it
            we can reach each node of the digraph.
        """

        g_len = len(list(self))

        nodeff = self.nodal_efficiency_kernel(list(self), g_len)
        nx.set_node_attributes(self, nodeff, name="nodal_eff")

    def local_efficiency_kernel(self, nodes):
        """

        Compute local efficiency, starting from nodal efficiency attribute.

        :param list nodes: list of nodes for which to compute the
            efficiency between them and all the other nodes

        :return: local efficiency dictionary keyed by node
        :rtype: dict
        """

        if not nx.get_node_attributes(self, "nodal_eff"):
            raise ValueError("No nodal efficiency attribute in the graph.")

        dict_loceff = {}

        for n in nodes:
            subgraph = list(self.successors(n))
            denom_subg = len(list(subgraph))

            if denom_subg != 0:
                sum_efficiencies = 0
                for w in list(subgraph):
                    kv_efficiency = self.nodes[w]["nodal_eff"]
                    sum_efficiencies = sum_efficiencies + kv_efficiency

                dict_loceff[n] = sum_efficiencies / denom_subg

            else:
                dict_loceff[n] = 0.

        return dict_loceff

    def local_efficiency(self):
        """

        Local efficiency of the node.

        .. note:: The local efficiency shows the efficiency of the connections
            between the first-order outgoing neighbors of node v
            when v is removed. Equivalently, local efficiency measures
            the "resilience" of the digraph to the perturbation of node removal,
            i.e. if we remove a node, how efficiently its first-order outgoing
            neighbors can communicate.
            It is in the range [0, 1].
        """

        loceff = self.local_efficiency_kernel(list(self))
        nx.set_node_attributes(self, loceff, name="local_eff")

    @property
    def global_efficiency(self):
        """

        Average global efficiency of the whole graph.

        .. note:: The average global efficiency of a graph is the average
            efficiency of all pairs of nodes.
        """

        g_len = len(list(self))

        if g_len <= 1:
            raise ValueError("Graph size must equal or larger than 2.")

        if not nx.get_node_attributes(self, "nodal_eff"):
            raise ValueError("No nodal efficiency attribute in the graph.")

        nodeff = nx.get_node_attributes(self, 'nodal_eff')
        nodeff_values = list(nodeff.values())

        return sum(nodeff_values) / g_len

    def shortest_path_list_kernel(self, nodes, tot_shortest_paths):
        """

        Collect the shortest paths that contain at least two nodes.

        :param list nodes: list of nodes for which to compute the
            list of shortest paths
        :param tot_shortest_paths: nested dictionary with key corresponding to
            source, while as value a dictionary keyed by target and valued
            by the source-target efficiency

        :return: list of shortest paths
        :rtype: list
        """

        tot_shortest_paths_list = list()

        for n in nodes:
            node_tot_shortest_paths = tot_shortest_paths[n]
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

        dict_betcen = {}

        for n in nodes:
            sp_with_node = []
            for l in tot_shortest_paths_list:
                if n in l and n != l[0] and n != l[-1]:
                    sp_with_node.append(l)

            numb_sp_with_node = len(sp_with_node)
            dict_betcen[n] = numb_sp_with_node / len(tot_shortest_paths_list)

        return dict_betcen

    def betweenness_centrality(self):
        """

        Betweenness_centrality measure of each node.
        Nodes' "betweenness_centrality" attribute is evaluated.

        .. note:: Betweenness centrality is an index of the relative importance
            of a node and it is defined by the number of shortest paths that run
            through it.
            Nodes with the highest betweenness centrality hold the higher level
            of control on the information flowing between different nodes in
            the network, because more information will pass through them.
        """

        if not nx.get_node_attributes(self, "shortest_path"):
            raise ValueError("No shortest path attribute in the graph.")

        tot_shortest_paths = nx.get_node_attributes(self, 'shortest_path')
        tot_shortest_paths_list = self.shortest_path_list_kernel(list(self),
            tot_shortest_paths)

        betcen = self.betweenness_centrality_kernel(list(self),
            tot_shortest_paths_list)
        nx.set_node_attributes(self, betcen, name="betweenness_centrality")

    def closeness_centrality_kernel(self, nodes, tot_shpaths_list, g_len):
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
            raise ValueError("Graph size must equal or larger than 2.")

        if not nx.get_node_attributes(self, "shpath_length"):
            raise ValueError("No shortest path length attribute in the graph.")

        dict_clocen = {}

        for n in nodes:
            totsp = []
            sp_with_node = []
            for l in tot_shpaths_list:
                if n in l and n == l[-1]:
                    sp_with_node.append(l)
                    length_path = self.nodes[l[0]]["shpath_length"][l[-1]]
                    totsp.append(length_path)
            norm = len(totsp) / (g_len - 1)

            if (sum(totsp)) != 0:
                dict_clocen[n] = (len(totsp) / sum(totsp)) * norm
            else:
                dict_clocen[n] = 0

        return dict_clocen

    def closeness_centrality(self):
        """

        Closeness_centrality measure of each node.
        Nodes' "closeness_centrality" attribute is evaluated.

        .. note:: Closeness centrality measures the reciprocal of the
            average shortest path distance from a node to all other reachable
            nodes in the graph. Thus, the more central a node is, the closer
            it is to all other nodes. This measure allows to identify good
            broadcasters, that is key elements in a graph, depicting how
            closely the nodes are connected with each other.
        """

        if not nx.get_node_attributes(self, "shortest_path"):
            raise ValueError("No shortest path attribute in the graph.")

        g_len = len(list(self))

        tot_shortest_paths = nx.get_node_attributes(self, 'shortest_path')
        tot_shortest_paths_list = self.shortest_path_list_kernel(list(self),
            tot_shortest_paths)

        clocen = self.closeness_centrality_kernel(list(self), \
        tot_shortest_paths_list, g_len)
        nx.set_node_attributes(self, clocen, name="closeness_centrality")

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
            raise ValueError("Graph size must equal or larger than 2.")

        dict_deg_cen = {}

        for n in nodes:
            num_neighbor_nodes = self.degree(n, weight = 'weight')
            dict_deg_cen[n] = num_neighbor_nodes / (g_len - 1)

        return dict_deg_cen

    def degree_centrality(self):
        """

        Degree centrality measure of each node.
        Nodes' "degree_centrality" attribute is evaluated.

        .. note:: Degree centrality is a simple centrality measure that counts
            how many neighbors a node has in an undirected graph.
            The more neighbors the node has the most important it is,
            occupying a strategic position that serves as a source or conduit
            for large volumes of flux transactions with other nodes.
            A node with high degree centrality is a node with many dependencies.
        """

        g_len = len(list(self))

        deg_cen = self.degree_centrality_kernel(list(self), g_len)
        nx.set_node_attributes(self, deg_cen, name="degree_centrality")

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
            raise ValueError("Graph size must equal or larger than 2.")

        dict_indeg_cen = {}

        for n in nodes:
            num_incoming_nodes = self.in_degree(n, weight = 'weight')
            dict_indeg_cen[n] = num_incoming_nodes / (g_len - 1)

        return dict_indeg_cen

    def indegree_centrality(self):
        """

        Indegree centrality measure of each node.
        Nodes' "indegree_centrality" attribute is evaluated.

        .. note:: Indegree centrality is measured by the number of edges
            ending at the node in a directed graph. Nodes with high indegree
            centrality are called cascade resulting nodes.
        """

        g_len = len(list(self))

        indeg_cen = self.indegree_centrality_kernel(list(self), g_len)
        nx.set_node_attributes(self, indeg_cen, name="indegree_centrality")

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
            raise ValueError("Graph size must equal or larger than 2.")

        dict_outdeg_cen = {}

        for n in nodes:
            num_outcoming_nodes = self.out_degree(n, weight = 'weight')
            dict_outdeg_cen[n] = num_outcoming_nodes / (g_len - 1)

        return dict_outdeg_cen

    def outdegree_centrality(self):
        """

        Outdegree centrality measure of each node.
        Nodes' "outdegree_centrality" attribute is evaluated.

        .. note:: Outdegree centrality is measured by the number of edges
            starting from a node in a directed graph. Nodes with high outdegree
            centrality are called cascade inititing nodes.
        """

        g_len = len(list(self))

        outdeg_cen = self.outdegree_centrality_kernel(list(self), g_len)
        nx.set_node_attributes(self, outdeg_cen, name="outdegree_centrality")

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

        print("In the graph are present", n_of_nodes, "nodes")
        print("go serial!")
        if g_density <= 0.000001:
            print("the graph is sparse, density =", g_density)
            self.dijkstra_single_source_shortest_path()
        else:
            print("the graph is dense, density =", g_density)
            self.floyd_warshall_predecessor_and_distance()

    def compute_service(self):
        """

        Compute service for every node,
        together with edge splitting.

        :param graph: Graph where the service is updated
        :type graph: networkx.DiGraph
        :param str servicename: service to populate
        """

        if not nx.get_node_attributes(self, "shortest_path"):
            raise ValueError("No shortest path attribute in the graph.")

        users_per_node = {node: 0. for node in self}
        splitting = {edge: 0. for edge in self.edges()}
        nx.set_node_attributes(self, 0., 'service')

        users_per_source = {
            s: [u for u in self.USER if nx.has_path(self, s, u)]
            for s in self.SOURCE
        }

        for s in self.SOURCE:
            for u in users_per_source[s]:
                for node in self.nodes[s]["shortest_path"][u]:
                    users_per_node[node] += 1.

        for s in self.SOURCE:
            for u in users_per_source[s]:
                self.nodes[u]['service'] += \
                self.service[s]/len(users_per_source[s])

        #Cycle just on the edges contained in source-user shortest paths
        for s in self.SOURCE:
            for u in users_per_source[s]:
                for idx in range(len(self.nodes[s]["shortest_path"][u])-1):

                    head = self.nodes[s]["shortest_path"][u][idx]
                    tail = self.nodes[s]["shortest_path"][u][idx+1]

                    splitting[(head, tail)] += 1./users_per_node[head]
