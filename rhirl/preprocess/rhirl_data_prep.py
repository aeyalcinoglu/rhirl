"""
The main setup file for RHIRL
"""


import ast
import torch
import osmnx as ox
from time import perf_counter
from rhirl.preprocess.rhirl_data_prep_utils import *
from rhirl.evaluation.rhirl_eval_utils import (get_segment_index_to_distance,
                                               get_traj_index_to_distance)
from rhirl.model.constants import *


def setup_initial():
    train_df = pd.read_csv(train_data_file_name)
    val_df = pd.read_csv(val_data_file_name)
    test_df = pd.read_csv(test_data_file_name)
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    train_length = len(train_df)
    val_length = len(val_df)
    val_start_index = train_length
    test_start_index = train_length + val_length

    # hour is a rounded version of the actual time
    # [x:30:00, x+1:29:59] -> x:00:00
    times = (pd.to_datetime(df["HOUR"]).astype('int64') // 10**9).values
    sorted_indices = np.argsort(times)
    df = df.iloc[sorted_indices]

    trajectories = [ast.literal_eval(trajectory)
                    for trajectory in df["NODES"]]
    coords_mm = [ast.literal_eval(coords_mm)
                 for coords_mm in df["COORDS_MM"]]

    train_times = times[:val_start_index]
    test_times = times[test_start_index:]

    dirty_road_segments = get_dirty_road_segments(
        trajectories)
    n = len(dirty_road_segments)
    clean_trajectories = get_clean_trajectories(
        dirty_road_segments, trajectories)
    train_trajectories = clean_trajectories[:val_start_index]
    test_trajectories = clean_trajectories[test_start_index:]

    integer_segment_to_coords = get_integer_segment_to_coords(
        coords_mm, clean_trajectories)

    graph = ox.graph_from_place('Porto, Portugal', network_type='drive')
    segment_punishments = get_segment_punishments(
        graph, integer_segment_to_coords)

    train_coords_mm = coords_mm[:val_start_index]
    test_coords_mm = coords_mm[test_start_index:]
    train_speeds = get_speeds(train_coords_mm)
    test_speeds = get_speeds(test_coords_mm)
    train_traffic_speeds = get_traffic_speeds(
        train_trajectories, train_speeds, train_times, n, default_speed_value)
    test_traffic_speeds = get_traffic_speeds(
        test_trajectories, test_speeds, test_times, n, default_speed_value)

    historical_trajectories = get_historical_trajectories(
        train_trajectories, train_times)
    adjacency_matrix = get_adjacency_matrix(n, clean_trajectories)
    real_svf = get_real_svf(n, historical_trajectories)
    train_finder_for_all_trajectories_between = get_finder_for_all_trajectories_between(
        train_trajectories)

    np.save(adjacency_matrix_file_name, adjacency_matrix)
    np.save(segment_punishments_file_name, segment_punishments)
    np.save(test_times_file_name, test_times)
    torch.save(real_svf, real_svf_file_name)

    torch.save(train_traffic_speeds, train_traffic_speeds_file_name)
    torch.save(test_traffic_speeds, test_traffic_speeds_file_name)

    with open(train_trajectories_file_name, 'wb') as file:
        pickle.dump(train_trajectories, file)
    with open(test_trajectories_file_name, 'wb') as file:
        pickle.dump(test_trajectories, file)

    with open(integer_segment_to_coords_file_name, 'wb') as file:
        pickle.dump(integer_segment_to_coords, file)
    with open(historical_trajectories_file_name, 'wb') as file:
        pickle.dump(historical_trajectories, file)
    with open(train_finder_for_all_trajectories_between_file_name, 'wb') as file:
        pickle.dump(train_finder_for_all_trajectories_between, file)


def setup_train(device):
    start_time = perf_counter()

    adjacency_matrix = np.load(adjacency_matrix_file_name)
    adjacency_matrix = torch.tensor(adjacency_matrix).to(device)

    if PUNISH:
        segment_punishments = np.load(segment_punishments_file_name)
        segment_punishments = torch.tensor(segment_punishments).view(1, -1, 1)
    else:
        segment_punishments = torch.ones(1, len(adjacency_matrix), 1)

    real_svf = torch.load(real_svf_file_name)
    with open(historical_trajectories_file_name, 'rb') as file:
        historical_trajectories = pickle.load(file)
    with open(train_finder_for_all_trajectories_between_file_name, 'rb') as file:
        train_finder_for_all_trajectories_between = pickle.load(file)

    train_traffic_speeds = torch.load(train_traffic_speeds_file_name)

    after_time = perf_counter()
    print('Train setup took {} seconds'.format(after_time - start_time))

    return historical_trajectories, adjacency_matrix, segment_punishments, train_traffic_speeds, real_svf, train_finder_for_all_trajectories_between


def setup_reward_generation(device):
    start_time = perf_counter()

    adjacency_matrix = np.load(adjacency_matrix_file_name)
    adjacency_matrix = torch.tensor(adjacency_matrix).to(device)

    test_traffic_speeds = torch.load(test_traffic_speeds_file_name)

    after_time = perf_counter()
    print('Reward generation setup took {} seconds'.format(after_time - start_time))

    return adjacency_matrix, test_traffic_speeds


def setup_eval(device):
    adjacency_matrix = np.load(adjacency_matrix_file_name)
    test_times = np.load(test_times_file_name)
    with open(integer_segment_to_coords_file_name, 'rb') as file:
        integer_segment_to_coords = pickle.load(file)
    with open(test_trajectories_file_name, 'rb') as file:
        test_trajectories = pickle.load(file)
    print('There are {} test cases'.format(len(test_trajectories)))

    n = len(adjacency_matrix)
    adjacency_matrix = torch.tensor(adjacency_matrix).to(device)
    segment_index_to_distance = get_segment_index_to_distance(
        n, integer_segment_to_coords)
    traj_index_to_distance = get_traj_index_to_distance(
        test_trajectories, segment_index_to_distance)

    return adjacency_matrix, test_times, test_trajectories, segment_index_to_distance, traj_index_to_distance


if __name__ == "__main__":
    """
    Setup necessary data
    See README and constants.py for more info
    """
    print("Doing the initial setup, should take around two minutes")
    start_time = perf_counter()
    setup_initial()
    after_time = perf_counter()
    print('Initial setup took {} seconds'.format(after_time - start_time))
