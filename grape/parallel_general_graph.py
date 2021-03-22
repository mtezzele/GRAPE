"""ParallelGeneralGraph for parallel directed graphs (DiGraph) module"""

import logging
import sys
import warnings
from multiprocessing import Queue
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import ctypes
import copy
import numpy as np
import networkx as nx
import pandas as pd

from .utils import chunk_it
from .general_graph import GeneralGraph

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class ParallelGeneralGraph(GeneralGraph):
    """
    Class ParallelGeneralGraph for parallel implementation of
    directed graphs (DiGraph).

    Constructs a new graph given an input file.
    A DiGraph stores nodes and edges with optional data or attributes.
    DiGraphs hold directed edges.
    Nodes can be arbitrary python objects with optional key/value attributes.
    Edges are represented  as links between nodes with optional key/value
    attributes.
    """

    def __init__(self):
        super().__init__()
        self.manager = mp.Manager()
        self.num = mp.cpu_count()

    def measure_iteration(self, nodes, record, kernel=None, *measure_args):
        """

        Inner iteration for parallel measures,
        to update shared dictionary.

        :param list nodes: nodes for which to compute the
            shortest path between them and all the other nodes
        :param multiprocessing.managers.dict record:
            shared dictionary to be updated
        :param callable kernel: kernel measure to be computed
        :param \*measure_args: arguments for kernel measures
           (have a look at specific kernel measures in 
           GeneralGraph for the particular variables/types
           for each measure)
        """

        if kernel is None:
            raise ValueError('No kernel function provided.')

        partial_dict = kernel(nodes, *measure_args)
        record.update(partial_dict)

    def measure_processes(self, record, kernel=None, *measure_args):
        """

        Division of total number of nodes in chuncks and
        parallel distribution of tasks into processes,
        for different kernel measure functions.

        :param callable kernel: kernel measure to be computed
        :param multiprocessing.managers.dict record:
            shared dictionary to be updated
        :param \*measure_args: arguments for kernel measures
           (have a look at specific kernel measures in GeneralGraph
           for the particular variables/types for each measure)
        """
        if kernel is None:
            raise ValueError('No kernel function provided.')

        node_chunks = chunk_it(list(self.nodes()), self.num)

        processes = [
            mp.Process( target=self.measure_iteration,
            args=(node_chunks[p], record, kernel, *measure_args) )
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

    def floyd_warshall_predecessor_and_distance(self):
        """

        Parallel Floyd Warshall's APSP algorithm. The predecessors
        and distance matrices are evaluated, together with the nested
        dictionaries for shortest-path, length of the paths and
        efficiency attributes.

        .. note:: Edges weight is taken into account in the distance matrix.
            Edge weight attributes must be numerical. Distances are calculated
            as sums of weighted edges traversed.
        """

        dist, pred = self.floyd_warshall_initialization()

        shared_d = mp.sharedctypes.RawArray(ctypes.c_double, dist.shape[0]**2)
        dist_shared = np.frombuffer(shared_d, 'float64').reshape(dist.shape)
        dist_shared[:] = dist

        shared_p = mp.sharedctypes.RawArray(ctypes.c_double,pred.shape[0]**2)
        pred_shared = np.frombuffer(shared_p, 'float64').reshape(pred.shape)
        pred_shared[:] = pred

        n = len(self.nodes())
        chunk = [(0, int(n / self.num))]
        node_chunks = chunk_it(list(self.nodes()), self.num)

        for i in range(1, self.num):
            chunk.append((chunk[i - 1][1],
                          chunk[i - 1][1] + len(node_chunks[i])))

        barrier = mp.Barrier(self.num)
        processes = [
            mp.Process( target=self.floyd_warshall_kernel,
            args=(dist_shared, pred_shared, chunk[p][0], chunk[p][1], barrier))
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        all_shortest_path = self.manager.dict()

        processes = [
            mp.Process( target=self.measure_iteration,
            args=(list(map(self.ids_reversed.get, node_chunks[p])),
                all_shortest_path, self.construct_path_kernel, pred_shared) )
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

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
                length_path = dist_shared[self.ids_reversed[value[0]],
                                          self.ids_reversed[value[-1]]]
                shortest_path_length[self.ids[i]][key] =  length_path
        
        return nonempty_shortest_path, shortest_path_length

    def dijkstra_iteration_parallel(self, out_q, nodes):
        """

        Parallel SSSP algorithm based on Dijkstra’s method.

        :param multiprocessing.queues.Queue out_q: multiprocessing queue
        :param list nodes: list of starting nodes from which the SSSP should be
            computed to every other target node in the graph

        .. note:: Edges weight is taken into account.
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.
        """

        for n in nodes:
            ssspp = (n, nx.single_source_dijkstra(self, n, weight='weight'))
            out_q.put(ssspp)

    def dijkstra_single_source_shortest_path(self):
        """

        Wrapper for parallel SSSP algorithm based on Dijkstra’s method.
        The nested dictionaries for shortest-path, length of the paths and
        efficiency attributes are evaluated.

        .. note:: Edges weight is taken into account.
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.
        """

        self.attribute_ssspp = []

        out_q = Queue()

        node_chunks = chunk_it(list(self.nodes()), self.num)

        processes = [
            mp.Process( target=self.dijkstra_iteration_parallel,
            args=( out_q,node_chunks[p] ))
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        while 1:
            running = any(p.is_alive() for p in processes)
            while not out_q.empty():

                self.attribute_ssspp.append(out_q.get())

            if not running:
                break

        shortest_path = {}
        shortest_path_length = {}
        for ssspp in self.attribute_ssspp:

            n = ssspp[0]
            shortest_path[n] = ssspp[1][1]
            shortest_path_length[n] = ssspp[1][0]

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

        logging.debug(f'Number of processors: {self.num}')

        logging.debug(f'In the graph are present {n_of_nodes} nodes')
        if g_density <= 0.000001:
            logging.debug(f'The graph is sparse, density = {g_density}')
            shpath, shpath_len = self.dijkstra_single_source_shortest_path()
        else:
            logging.debug(f'The graph is dense, density = {g_density}')
            shpath, shpath_len = self.floyd_warshall_predecessor_and_distance()

        return shpath, shpath_len

    def compute_efficiency(self):
        """

        Efficiency calculation.

        .. note:: The efficiency of a path connecting two nodes is defined
            as the inverse of the path length, if the path has length non-zero,
            and zero otherwise.
        """

        shortest_path_length = self.shortest_path_length
        efficiency = self.manager.dict()
        self.measure_processes(efficiency, self.efficiency_kernel, 
            shortest_path_length)
        return efficiency

    def compute_nodal_efficiency(self):
        """

        Nodal efficiency calculation.

        .. note:: The nodal efficiency of the node is equal to zero
            for a node without any outgoing path and equal to one if from it
            we can reach each node of the digraph.
        """

        g_len = len(list(self))
        efficiency = self.efficiency
        nod_eff = self.manager.dict()
        self.measure_processes(nod_eff, self.nodal_efficiency_kernel,
            efficiency, g_len)
        return nod_eff

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
        loc_eff = self.manager.dict()
        self.measure_processes(loc_eff, self.local_efficiency_kernel, 
            nodal_efficiency)
        return loc_eff

    def shortest_path_list_iteration(self, nodes, shpath, tot_shpaths_list):
        """

        Inner iteration for parallel shortest path list calculation,
        to update shared list.

        :param list nodes: list of nodes for which to compute the
            shortest path between them and all the other nodes
        :param tot_shpaths_list: list of shortest paths
            with at least two nodes
        :type tot_shpath_list: multiprocessing.managers.list
        """

        partial_shpath_list = self.shortest_path_list_kernel(nodes, shpath)
        tot_shpaths_list.extend(partial_shpath_list)

    def compute_betweenness_centrality(self):
        """

        Betweenness_centrality calculation.

        .. note:: Betweenness centrality is an index of the relative importance
            of a node and it is defined by the number of shortest paths that run
            through it.
            Nodes with the highest betweenness centrality hold the higher level
            of control on the information flowing between different nodes in
            the network, because more information will pass through them.
        """

        shortest_path = self.shortest_path
        tot_shortest_paths_list = self.manager.list()
        node_chunks = chunk_it(list(self.nodes()), self.num)

        processes = [
            mp.Process( target=self.shortest_path_list_iteration,
            args=(node_chunks[p], shortest_path, tot_shortest_paths_list) )
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        bet_cen = self.manager.dict()
        self.measure_processes(bet_cen, self.betweenness_centrality_kernel,
            tot_shortest_paths_list)
        
        return bet_cen

    def compute_closeness_centrality(self):
        """

        Closeness_centrality calculation.

        .. note:: Closeness centrality measures the reciprocal of the
            average shortest path distance from a node to all other reachable
            nodes in the graph. Thus, the more central a node is, the closer
            it is to all other nodes. This measure allows to identify good
            broadcasters, that is key elements in a graph, depicting how
            closely the nodes are connected with each other.
        """

        g_len = len(list(self))
        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        shortest_path = self.shortest_path
        shortest_path_length = self.shortest_path_length
        tot_shortest_paths_list = self.manager.list()
        node_chunks = chunk_it(list(self.nodes()), self.num)

        processes = [
            mp.Process( target=self.shortest_path_list_iteration,
            args=(node_chunks[p], shortest_path, tot_shortest_paths_list) )
            for p in range(self.num) ]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        clo_cen = self.manager.dict()
        self.measure_processes(clo_cen, self.closeness_centrality_kernel,
            shortest_path_length, tot_shortest_paths_list, g_len)

        return clo_cen

    def compute_degree_centrality(self):
        """

        Degree centrality calculation.

        .. note:: Degree centrality is a simple centrality measure that counts
            how many neighbors a node has in an undirected graph.
            The more neighbors the node has the most important it is,
            occupying a strategic position that serves as a source or conduit
            for large volumes of flux transactions with other nodes.
            A node with high degree centrality is a node with many dependencies.
        """

        g_len = len(list(self))
        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        deg_cen = self.manager.dict()
        self.measure_processes(deg_cen, self.degree_centrality_kernel, g_len)
        return deg_cen

    def compute_indegree_centrality(self):
        """

        In-degree centrality calculation.

        .. note:: In-degree centrality is measured by the number of edges
            ending at the node in a directed graph. Nodes with high in-degree
            centrality are called cascade resulting nodes.
        """

        g_len = len(list(self))
        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        indeg_cen = self.manager.dict()
        self.measure_processes(indeg_cen, self.indegree_centrality_kernel,
            g_len)
        return indeg_cen

    def compute_outdegree_centrality(self):
        """

        Out-degree centrality calculation.

        .. note:: Out-degree centrality is measured by the number of edges
            starting from a node in a directed graph. Nodes with high out-degree
            centrality are called cascade inititing nodes.
        """

        g_len = len(list(self))
        if g_len <= 1:
            raise ValueError('Graph size must equal or larger than 2.')

        outdeg_cen = self.manager.dict()
        self.measure_processes(outdeg_cen, self.outdegree_centrality_kernel,
            g_len)
        return outdeg_cen
