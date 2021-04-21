ParallelGeneralGraph
=====================

.. currentmodule:: grape.parallel_general_graph

.. automodule:: grape.parallel_general_graph

.. autosummary::
    :toctree: _summaries
    :nosignatures:

    ParallelGeneralGraph
    ParallelGeneralGraph.measure_iteration
    ParallelGeneralGraph.measure_processes
    ParallelGeneralGraph.floyd_warshall_predecessor_and_distance
    ParallelGeneralGraph.dijkstra_iteration_parallel
    ParallelGeneralGraph.dijkstra_single_source_shortest_path
    ParallelGeneralGraph.calculate_shortest_path
    ParallelGeneralGraph.compute_efficiency
    ParallelGeneralGraph.compute_nodal_efficiency
    ParallelGeneralGraph.compute_local_efficiency
    ParallelGeneralGraph.shortest_path_list_iteration
    ParallelGeneralGraph.compute_betweenness_centrality
    ParallelGeneralGraph.compute_closeness_centrality
    ParallelGeneralGraph.compute_degree_centrality
    ParallelGeneralGraph.compute_indegree_centrality
    ParallelGeneralGraph.compute_outdegree_centrality

.. autoclass:: ParallelGeneralGraph
    :members: 
    :inherited-members: load, mark, area, perturbation_resistant, description,
        init_status, final_status, mark_status, status_area, father_condition,
        weight, type, sources, users, switches, initial_service, service,
        shortest_path, shortest_path_length, efficiency, nodal_efficiency,
        local_efficiency, global_efficiency, betweenness_centrality,
        closeness_centrality, degree_centrality, indegree_centrality,
        outdegree_centrality, clear_data, construct_path_kernel,
        floyd_warshall_initialization, floyd_warshall_kernel,
        efficiency_kernel, nodal_efficiency_kernel, local_efficiency_kernel,
        shortest_path_list_kernel, betweenness_centrality_kernel,
        closeness_centrality_kernel, degree_centrality_kernel,
        indegree_centrality_kernel, outdegree_centrality_kernel, compute_service
    :private-members:
    :undoc-members:
    :show-inheritance: GeneralGraph
    :noindex:
