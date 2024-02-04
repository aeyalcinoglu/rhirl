"""
This file contains some functions to play with graphs,
line graphs and random walks
See https://www.yalcinoglu.com/assets/dynamics_line_digraphs.pdf
"""


import numpy as np
from collections import defaultdict
import random


def get_random_directed_adj_matrix(vertex_count, edge_count):
    """
    To get a more realistic graph, use
    networkx.generators.random_graphs.erdos_renyi_graph.html
    One can also take the strongly connected component via
    networkx.algorithms.components.strongly_connected_components.html
    """
    assert edge_count >= vertex_count - 1
    assert edge_count <= vertex_count * (vertex_count - 1)

    adj_matrix = [[0] * vertex_count for _ in range(vertex_count)]
    edges = [(i, j) for i in range(vertex_count)
             for j in range(vertex_count) if i != j]
    random.shuffle(edges)

    for _ in range(edge_count):
        u, v = edges.pop()
        adj_matrix[u][v] = 1

    return adj_matrix


def directed_random_walk(adj_matrix, iteration_count, walk_length):
    """
    Returns trajectories of random walks on a directed graph
    `iteration_count` many `walk_length` long random walks
    """
    np_adjacency_matrix = np.array(adj_matrix)
    num_nodes = len(np_adjacency_matrix)
    trajectories = []

    for _ in range(iteration_count):
        start_node = np.random.randint(num_nodes)
        trajectory = [start_node]

        for _ in range(walk_length - 1):
            current_node = trajectory[-1]
            neighbors = np.where(np_adjacency_matrix[current_node] > 0)[0]
            if len(neighbors) == 0:
                break
            next_node = np.random.choice(neighbors)
            trajectory.append(next_node)

        trajectories.append(trajectory)

    return trajectories


def get_directed_line_matrix(adjacency_matrix):
    n = int(np.sum(np.sum(adjacency_matrix)))
    line_adj_matrix = np.zeros((n, n), dtype=int)
    unique_edges = set()

    for i in range(0, len(adjacency_matrix)):
        for j in range(0, len(adjacency_matrix)):
            if adjacency_matrix[i][j] == 1:
                unique_edges.add((i, j))

    assert len(unique_edges) == n
    unique_edges = list(unique_edges)

    for i, unique_edge in enumerate(unique_edges):
        for j, unique_edge2 in enumerate(unique_edges):
            if unique_edge != unique_edge2 and unique_edge[1] == unique_edge2[0]:
                line_adj_matrix[i][j] = 1

    return unique_edges, line_adj_matrix.tolist()


def rsvf_on_edges(unique_edges, trajectories):
    rsvf = {}
    for traj in trajectories:
        for i in range(len(traj)-1):
            edge_index = unique_edges.index((traj[i], traj[i+1]))
            if edge_index not in rsvf.keys():
                rsvf[edge_index] = 1
            else:
                rsvf[edge_index] += 1
    return rsvf


def rsvf_on_verticies(trajectories):
    rsvf = defaultdict(int)
    for traj in trajectories:
        for node in traj:
            rsvf[node] += 1
    return rsvf


if __name__ == "__main__":
    vertex_count, edge_count, iteration_count, walk_length = 7, 20, 10000, 100
    normal_adj_matrix = get_random_directed_adj_matrix(
        vertex_count, edge_count)
    unique_edges, line_adj_matrix = get_directed_line_matrix(normal_adj_matrix)
    normal_trajectories = directed_random_walk(normal_adj_matrix,
                                               iteration_count,
                                               walk_length)
    line_trajectories = directed_random_walk(line_adj_matrix,
                                             iteration_count,
                                             walk_length)
    normal_rsvf = sorted(
        list(rsvf_on_edges(unique_edges, normal_trajectories).values()))
    line_rsvf = sorted(list(rsvf_on_verticies(line_trajectories).values()))

    print("Edge freqs on G: {}".format(
        [round(rsvf / sum(normal_rsvf), 2) for rsvf in normal_rsvf]))
    print("Vertex freqs on L(G): {}".format(
        [round(rsvf / sum(line_rsvf), 2) for rsvf in line_rsvf]))
