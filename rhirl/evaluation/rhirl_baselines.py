"""
This file contains the implementation of the baselines for comparison with RHIRL
"""


from collections import Counter
import networkx as nx
from rhirl.evaluation.rhirl_eval_utils import get_networkx_graph_for_dijkstra


def MPR(test_trajectories, train_finder):
    """
    Most popular route
    Given a trajectory, find the most common trajectory
    Among the trajectories that has the same start and end
    """
    model_trajectories = []
    cached_results = {}

    for trajectory in test_trajectories:
        start, end = trajectory[0], trajectory[-1]

        if (start, end) not in cached_results.keys():
            same_trajs = train_finder.get((start, end), [])

            if len(same_trajs) == 0:
                cached_results[(start, end)] = []

            else:
                traj_counter = Counter(map(tuple, same_trajs))
                most_common_traj = max(traj_counter, key=traj_counter.get)
                cached_results[(start, end)] = list(most_common_traj)

        model_trajectories.append(cached_results[(start, end)])

    return model_trajectories


def Dijkstra(test_trajectories, adjacency_matrix, segment_index_to_distance):
    """
    It assumes that the first and last road segment is correctly predicted
    as opposed to first and last node of a trajectory
    """
    graph = get_networkx_graph_for_dijkstra(
        adjacency_matrix, segment_index_to_distance)

    model_trajectories = []
    cached_results = {}

    for trajectory in test_trajectories:
        origin, destination = trajectory[0], trajectory[-1]
        if (origin, destination) not in cached_results.keys():
            shortest_path = nx.dijkstra_path(
                graph, origin, destination, weight='weight')
            cached_results[(origin, destination)] = shortest_path
        model_trajectories.append(cached_results[(origin, destination)])

    return model_trajectories
