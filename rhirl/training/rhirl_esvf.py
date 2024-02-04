"""
This file contains the function to compute the expected svf
and the function to compute the difference between the real svf and the expected svf
"""


import torch
import torch.nn as nn


def get_expected_svf(n, historical_trajectories,
                     finder_for_all_trajectories_between,
                     rewards,
                     time):
    esvf = torch.zeros(n, dtype=torch.float32)

    for trajectory in historical_trajectories[time]:
        l_o, l_d = trajectory[0], trajectory[-1]
        if l_o == l_d:
            continue
        # does adding one trajectory deserve the time? I don't think so
        # eta = dijkstra(adjacency_matrix, weights, l_o, l_d)[0]
        # if eta not in H_trajectory:
        #     H_trajectories.append(eta)

        H_trajectories = finder_for_all_trajectories_between[(l_o, l_d)]
        R_trajectories = [rewards[H_trajectory].sum()
                          for H_trajectory in H_trajectories]
        soft_R = nn.functional.softmax(torch.tensor(
            R_trajectories, dtype=torch.float32), dim=0)

        for i, H_trajectory in enumerate(H_trajectories):
            esvf[H_trajectory] += soft_R[i].item()

    return esvf


def get_scales_batch(n,
                     rewards_batch,
                     historical_trajectories,
                     real_svf,
                     finder_for_all_trajectories_between,
                     time_batch):
    """
    Returns the difference between the real svf and the expected svf
    """
    scales_list = []

    for delta, time in enumerate(time_batch):
        time = int(time)

        current_rewards = rewards_batch[delta]
        current_rsvf = real_svf[time]

        esvf = get_expected_svf(n, historical_trajectories,
                                finder_for_all_trajectories_between,
                                current_rewards, time)
        scales_list.append(current_rsvf - esvf)

    scales_batch = torch.stack(scales_list, dim=0).unsqueeze(2)

    return scales_batch
