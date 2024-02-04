"""
This file contains some utility functions for preprocessing for RHIRL
"""


from collections import defaultdict
import math
import pickle
import numpy as np
import pandas as pd
import osmnx as ox
import torch
from rhirl.model.constants import *


def defaultdict_for_int():
    """
    Because pickle can't handle lambdas
    It was a fix provided in StackOverflow
    """
    return defaultdict(int)


def haversine(coord1, coord2):
    R = 6371000  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * \
        math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_dirty_road_segments(trajectories):
    """
    A road segment is a pair of nodes
    """
    road_segments = set()
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            road_segment = (trajectory[i], trajectory[i+1])
            road_segments.add(road_segment)

    road_segments = list(road_segments)
    road_segments.sort(key=lambda pair: (pair[0], pair[1]))
    road_segments = np.array(road_segments)

    return road_segments


def get_clean_trajectories(dirty_road_segments, trajectories):
    """
    A clean trajectory is a list of integers,
    each integer corresponds to a road segment
    """
    segment_to_index = {tuple(segment): idx for idx,
                        segment in enumerate(dirty_road_segments)}

    clean_trajectories = []

    for trajectory in trajectories:
        clean_trajectory = []
        for j in range(len(trajectory) - 1):
            current_segment = (trajectory[j], trajectory[j+1])
            road_segment_index = segment_to_index.get(current_segment)
            clean_trajectory.append(road_segment_index)
        clean_trajectories.append(clean_trajectory)

    return clean_trajectories


def get_integer_segment_to_coords(coords_mm, clean_trajectories):
    integer_segment_to_coords = {}
    for i, trajectory in enumerate(clean_trajectories):
        for j, road_segment in enumerate(trajectory):
            if road_segment not in integer_segment_to_coords.keys():
                integer_segment_to_coords[road_segment] = [
                    coords_mm[i][j], coords_mm[i][j+1]]

    return integer_segment_to_coords


def get_osm_node_index(graph, node):
    osm_node = ox.distance.nearest_nodes(graph, node[1], node[0])
    match_problem = [graph.nodes[osm_node]['y'],
                     graph.nodes[osm_node]['x']] != node

    if match_problem:
        return False

    return osm_node


def get_segment_punishments(graph, integer_segment_to_coords):
    """
    Amplify the rewards of the segments which are
    consistent with OSM, see the report for more details
    """
    osm_node_index = {}
    segment_punishments = np.zeros(
        len(integer_segment_to_coords), dtype=np.uint8)
    for i, segment in integer_segment_to_coords.items():
        if tuple(segment[0]) not in osm_node_index.keys():
            osm_node_index[tuple(segment[0])] = get_osm_node_index(
                graph, segment[0])
        if tuple(segment[1]) not in osm_node_index.keys():
            osm_node_index[tuple(segment[1])] = get_osm_node_index(
                graph, segment[1])
        if not osm_node_index[tuple(segment[0])] or not osm_node_index[tuple(segment[1])]:
            segment_punishments[i] = 1
            continue
        if not graph.has_edge(osm_node_index[tuple(segment[0])], osm_node_index[tuple(segment[1])]):
            segment_punishments[i] = 1
            continue
        segment_punishments[i] = 2

    return np.array(segment_punishments)


def get_speeds_in_trajectory(trajectory):
    speeds_in_trajectory = []

    for i in range(0, len(trajectory) - 1):
        speed_candidate = 3.6 * haversine(trajectory[i], trajectory[i+1]) / 15
        speeds_in_trajectory.append(max(minimum_speed,
                                        min(speed_candidate, maximum_speed)))

    return speeds_in_trajectory


def get_speeds(coords_mm):
    speeds = []
    for i in range(len(coords_mm)):
        speeds.append(get_speeds_in_trajectory(coords_mm[i]))
    return speeds


def get_traffic_speeds(trajectories, speeds, times, n, default_value):
    """
    Assumes 15 second between each road segment
    It is the most crude approximation
    Look at rhirl_speed.py for a more sophisticated approach
    """
    avg_speeds_dict = defaultdict(defaultdict_for_int)
    count = defaultdict(defaultdict_for_int)

    for i, trajectory in enumerate(trajectories):
        time = times[i]
        for j, road_segment in enumerate(trajectory):
            avg_speeds_dict[time][road_segment] += speeds[i][j]
            count[time][road_segment] += 1

    avg_speeds_tensor_dict = {}
    for time, road_segments_speeds in avg_speeds_dict.items():
        time_values = [road_segments_speeds[rs] / count[time][rs]
                       if count[time][rs] != 0 else default_value for rs in range(n)]
        avg_speeds_tensor_dict[time] = torch.tensor(time_values)

    avg_speeds_tensor_dict = {
        k: avg_speeds_tensor_dict[k] for k in sorted(avg_speeds_tensor_dict)}

    return avg_speeds_tensor_dict


def get_historical_trajectories(trajectories, times):
    """
    This is not H_tau, see find_valid_trajectories for that
    This returns a aggregated dictionary of trajectories
    Aggregated by time
    """
    historical_trajectories = defaultdict(list)
    for i in range(len(trajectories)):
        time = times[i]
        trajectory = trajectories[i]
        historical_trajectories[time].append(trajectory)
    historical_trajectories = {
        k: historical_trajectories[k] for k in sorted(historical_trajectories)}

    return historical_trajectories


def get_adjacency_matrix(n, trajectories):
    """
    As said in README.md, this gives a different result
    compared to OSM
    """
    adjacency_matrix = np.zeros((n, n), dtype=np.uint8)

    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            adjacency_matrix[trajectory[i], trajectory[i + 1]] = 1

    return adjacency_matrix


def get_real_svf(n, historical_trajectories):
    """
    Get real state visitation frequencies
    """
    rsvf = defaultdict(defaultdict_for_int)

    for time, trajectories in historical_trajectories.items():
        for trajectory in trajectories:
            for road_segment in trajectory:
                rsvf[time][road_segment] += 1.0
        # old mistake!
        # total = sum(rsvf[time].values())
        # rsvf[time] = {road_segment: value /
        #               total for road_segment, value in rsvf[time].items()}

    rsvf_tensor = {}
    for time, road_segments_rsvf in rsvf.items():
        time_values = [road_segments_rsvf[rs] for rs in range(n)]
        rsvf_tensor[time] = torch.tensor(time_values)

    rsvf_tensor = {k: rsvf_tensor[k] for k in sorted(rsvf_tensor)}

    return rsvf_tensor


def get_index_to_find_all_trajectories_between(trajectories):
    segment_to_trajectory = {}
    for i, trajectory in enumerate(trajectories):
        for segment in trajectory:
            if segment not in segment_to_trajectory:
                segment_to_trajectory[segment] = set()
            segment_to_trajectory[segment].add(i)

    return segment_to_trajectory


def find_valid_trajectories(trajectories, index_to_find_all_trajectories_between, s, e):
    """
    Argubly the core idea of the paper
    Generate trajectories to explore given a start and end
    Take all trajectories which have a path between s and e
    Truncate the trajectories to only include the path between s and e
    """
    valid_trajectories = []
    candidate_indices = index_to_find_all_trajectories_between.get(
        s, set()).intersection(index_to_find_all_trajectories_between.get(e, set()))

    for i in candidate_indices:
        trajectory = trajectories[i]
        s_found = False
        for j, segment in enumerate(trajectory):
            if segment == s:
                s_found = j
            elif segment == e and type(s_found) == int:
                valid_trajectories.append(trajectory[s_found:j+1])
                break

    return valid_trajectories


def get_finder_for_all_trajectories_between(trajectories):
    np_trajectories = np.array(trajectories, dtype=object)
    index_mapping = get_index_to_find_all_trajectories_between(np_trajectories)

    unique_start_ends = {(traj[0], traj[-1])
                         for traj in np_trajectories if traj[0] != traj[-1]}
    finder_for_all_trajectories_between = {pair: find_valid_trajectories(
        np_trajectories, index_mapping, *pair) for pair in unique_start_ends}

    return finder_for_all_trajectories_between
