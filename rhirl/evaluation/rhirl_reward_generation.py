"""
This file runs the saved model on test data to generate rewards
"""


from torch.utils.data import DataLoader
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm
from rhirl.model.rhirl import RHIRL, RHIRLDataset
from rhirl.preprocess.rhirl_data_prep import setup_reward_generation
from rhirl.model.constants import *


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    adjacency_matrix, test_traffic_speeds = setup_reward_generation(device)

    # change this name to the model you want to evaluate
    model_name = "2048-64-256-0.001_2024-01-25_17-19-57"
    saved_model_file_name = saved_model_params_dir_name + model_name + '.pt'
    test_rewards_file_name = saved_model_rewards_dir_name + model_name + ".pickle"

    n = len(adjacency_matrix)
    input_dim = n
    hidden_size, feature_dim, output_dim = map(
        int, model_name.split("_")[0].split("-")[:3])
    print("There are {} road segments".format(n))

    rhirl = RHIRL(input_dim, hidden_size, feature_dim,
                  output_dim, adjacency_matrix, TWO_HEAD)
    rhirl.to(device)
    rhirl.eval()
    rhirl.load_state_dict(torch.load(saved_model_file_name))
    test_dataset = RHIRLDataset(test_traffic_speeds)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    all_test_rewards = []
    for x_batch, y_batch, _ in tqdm(test_dataloader, desc="Testing", ncols=100):
        with torch.no_grad():
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            rewards_batch = rhirl(x_batch, y_batch)
            all_test_rewards.append(rewards_batch)

    test_rewards = {}
    available_times = list(test_traffic_speeds.keys())
    for i, test_reward in enumerate(all_test_rewards):
        for j, scene_reward in enumerate(test_reward):
            time_index = i * batch_size + j
            real_time = available_times[time_index]
            test_rewards[real_time] = scene_reward

    with open(test_rewards_file_name, 'wb') as file:
        pickle.dump(test_rewards, file)
