"""
This file contains the evaluation of RHIRL and baseline models
"""


import numpy as np
import pandas as pd
import torch
import json
import pickle
import random
from collections import defaultdict
from rhirl.preprocess.rhirl_data_prep import setup_eval
from rhirl.evaluation.rhirl_baselines import *
from rhirl.evaluation.rhirl_eval_utils import (
    get_rhirl_instance,
    categorize_by_time,
    fprecision, frecall,
    unweighted_precision,
    unweighted_recall,
    ff1_score)
from rhirl.model.constants import *


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    # change this name to the model you want to evaluate
    model_name = "2048-64-256-0.001_2024-01-25_17-19-57"
    # rhirl, mpr, dijkstra
    models_to_eval = ['rhirl']

    DISTANCE_WEIGHTED = False
    eval_results_file_name = saved_model_evals_dir_name + model_name + '.json'
    test_rewards_file_name = saved_model_rewards_dir_name + model_name + ".pickle"

    adjacency_matrix, test_times, test_trajectories, segment_index_to_distance, traj_index_to_distance = setup_eval(
        device)

    timed_indices = {7: [], 10: [], 17: [], 19: [], 24: []}
    for i, time in enumerate(test_times):
        hour = pd.to_datetime(time, unit='s').hour
        timed_indices[categorize_by_time(hour)].append(i)

    distanced_indices = {
        'short': [i for i, traj in enumerate(test_trajectories) if traj_index_to_distance[i] < 2],
        'long': [i for i, traj in enumerate(test_trajectories) if traj_index_to_distance[i] >= 2]
    }

    # get model results
    model_trajectory_list = []
    for name in models_to_eval:
        if name == 'rhirl':
            with open(test_rewards_file_name, 'rb') as file:
                test_rewards = pickle.load(file)
            model_trajectory_list.append(get_rhirl_instance(
                test_trajectories, test_times, adjacency_matrix, test_rewards))
        elif name == 'mpr':
            with open(train_finder_for_all_trajectories_between_file_name, 'rb') as file:
                train_finder_for_all_trajectories_between = pickle.load(file)
            model_trajectory_list.append(MPR(
                test_trajectories, train_finder_for_all_trajectories_between))
        elif name == 'dijkstra':
            model_trajectory_list.append(
                Dijkstra(test_trajectories, adjacency_matrix, segment_index_to_distance))
    eval_results = []

    # get evaluation results
    for model_trajectories in model_trajectory_list:
        results = defaultdict(list)

        for idx, (real_traj, model_traj) in enumerate(zip(test_trajectories, model_trajectories)):
            if DISTANCE_WEIGHTED:
                precision = fprecision(real_traj, model_traj,
                                       segment_index_to_distance)
                recall = frecall(real_traj, model_traj,
                                 segment_index_to_distance)
            else:
                precision = unweighted_precision(real_traj, model_traj)
                recall = unweighted_recall(real_traj, model_traj)
            f1_score = ff1_score(precision, recall)

            for category in ['short', 'long']:
                if idx in distanced_indices[category]:
                    results[category].append(
                        [precision, recall, f1_score])
            for time_cat in timed_indices:
                if idx in timed_indices[time_cat]:
                    results[time_cat].append(
                        [precision, recall, f1_score])

        total_results = results['short'] + results['long']
        averaged_total_results = [sum(vals) / len(vals)
                                  for vals in zip(*total_results)]

        averaged_results = {k: [sum(vals) / len(vals)
                                for vals in zip(*v)] for k, v in results.items()}

        averaged_results['total'] = averaged_total_results
        eval_results.append(averaged_results)

    with open(eval_results_file_name, 'w') as file:
        json.dump(eval_results, file, indent=4)
