"""Utility functions module"""


def chunk_it(nodes, n):
    """

    Divide nodes in chunks according to number of processes.

    :param list nodes: list of nodes
    :param int n: number of available processes
    
    :return: list of graph nodes to be assigned to every process
    :rtype: list
    """

    avg = len(nodes) / n
    out = []
    last = 0.0

    while last < len(nodes):
        out.append(nodes[int(last):int(last + avg)])
        last += avg
    return out
