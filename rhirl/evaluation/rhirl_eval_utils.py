"""
This file contains bunch of utility functions for evaluation
"""


import networkx as nx
import torch
from collections import defaultdict
from tqdm import tqdm
from rhirl.preprocess.rhirl_data_prep_utils import haversine


def create_graph_from_tensor(adj_matrix, weight_matrix):
    G = nx.DiGraph()
    rows, cols = torch.where(adj_matrix == 1)
    edges = [(row.item(), col.item(), {
              'weight': weight_matrix[row, col].item()}) for row, col in zip(rows, cols)]
    G.add_edges_from(edges)
    return G


def get_networkx_graph_for_dijkstra(adjacency_matrix, segment_index_to_distance):
    tensor_segment_index_to_distance = torch.tensor(
        segment_index_to_distance).unsqueeze(dim=-1).T.to(
        device=adjacency_matrix.device)
    distance_weights = tensor_segment_index_to_distance * adjacency_matrix
    distance_weights[distance_weights == 0] = float('inf')
    G = create_graph_from_tensor(adjacency_matrix, distance_weights)

    return G


def get_networkx_graph_for_rhirl(adjacency_matrix, rewards):
    # weight of an edge is the negative of the reward
    # of the segment that is at the end of the edge
    weights = (rewards * -1).T * adjacency_matrix
    weights[weights == 0] = float('inf')
    G = create_graph_from_tensor(adjacency_matrix, weights)

    return G


def get_segment_index_to_distance(n, integer_segment_to_coords):
    segment_index_to_distance = [None] * n
    for idx in range(n):
        segment = integer_segment_to_coords[idx]
        coord1, coord2 = segment[0], segment[1]
        segment_index_to_distance[idx] = haversine(coord1, coord2) / 1000

    return segment_index_to_distance


def get_traj_index_to_distance(test_trajectories, segment_index_to_distance):
    traj_index_to_distance = {}
    for idx, traj in enumerate(test_trajectories):
        total = 0
        for segment in traj:
            total += segment_index_to_distance[segment]
        traj_index_to_distance[idx] = total

    return traj_index_to_distance


def get_rhirl_instance(test_trajectories, test_times,
                       adjacency_matrix, test_rewards):
    # networkx is faster on cpu for shortest path
    test_rewards_cpu = {k: v.cpu() for k, v in test_rewards.items()}
    adjacency_matrix_cpu = adjacency_matrix.cpu()
    grouped_test_trajectories = defaultdict(list)
    for i, trajectory in enumerate(test_trajectories):
        time = test_times[i]
        grouped_test_trajectories[time].append((trajectory, i))

    model_trajectories = [None] * len(test_trajectories)
    for time in tqdm(grouped_test_trajectories.keys(), desc="Dijkstra(RHIRL EVAL)", ncols=100):
        G = get_networkx_graph_for_rhirl(
            adjacency_matrix_cpu, test_rewards_cpu[time])
        for trajectory, i in grouped_test_trajectories[time]:
            origin, destination = trajectory[0], trajectory[-1]
            model_trajectories[i] = nx.dijkstra_path(
                G, origin, destination, weight='weight')

    return model_trajectories


def fprecision(real_trajectory, model_trajectory, segment_index_to_distance):
    total_model_distance = 0
    correct_distance = 0

    for segment in model_trajectory:
        segment_distance = segment_index_to_distance[segment]
        total_model_distance += segment_distance
        if segment in real_trajectory:
            correct_distance += segment_distance

    return correct_distance / total_model_distance if total_model_distance else 0


def frecall(real_trajectory, model_trajectory, segment_index_to_distance):
    correct_distance = 0

    for segment in model_trajectory:
        if segment in real_trajectory:
            correct_distance += segment_index_to_distance[segment]
    total_real_distance = 0

    for segment in real_trajectory:
        total_real_distance += segment_index_to_distance[segment]

    return correct_distance / total_real_distance


def ff1_score(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0


def unweighted_precision(real_trajectory, model_trajectory):
    model_set = set(model_trajectory)
    real_set = set(real_trajectory)
    intersection = model_set.intersection(real_set)

    return len(intersection) / len(model_set) if model_set else 0


def unweighted_recall(real_trajectory, model_trajectory):
    model_set = set(model_trajectory)
    real_set = set(real_trajectory)
    intersection = model_set.intersection(real_set)

    return len(intersection) / len(real_set) if real_set else 0


def categorize_by_time(hour):
    if hour <= 7:
        return 7
    elif hour <= 10:
        return 10
    elif hour <= 17:
        return 17
    elif hour <= 19:
        return 19
    else:
        return 24
