"""
This file contains the training loop for RHIRL.
Setups the necessary data, prepares the model for training,
trains the model, saves tensorboard logs and saves the model.
"""


from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from rhirl.preprocess.rhirl_data_prep import setup_train
from rhirl.model.rhirl import RHIRL, RHIRLDataset
from rhirl.training.rhirl_esvf import get_scales_batch
from rhirl.model.constants import *


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    # Import train data, see constants.py for more info
    historical_trajectories, adjacency_matrix, segment_punishments, train_traffic_speeds, real_svf, finder_for_all_trajectories_between = setup_train(
        device)

    # Training parameters, see constants.py for more info
    number_of_road_segments = len(adjacency_matrix)
    input_dim = number_of_road_segments
    model_type = "{}-{}-{}-{}-{}".format(hidden_size,
                                         feature_dim, output_dim, lr)
    model_name = model_type + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tb_writer = SummaryWriter(training_logs_dir_name + model_name)
    model_save_path = saved_model_params_dir_name + model_name + ".pt"

    # Prepare the model for training
    print("OSM Punishments: {}".format(PUNISH))
    print("Two Head: {}".format(TWO_HEAD))
    rhirl = RHIRL(input_dim, hidden_size, feature_dim,
                  output_dim, adjacency_matrix, TWO_HEAD).to(device)
    rhirl.train()
    optimizer = torch.optim.Adam(rhirl.parameters(), lr=lr)
    train_dataset = RHIRLDataset(train_traffic_speeds)
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        for i, (x_batch, y_batch, time_batch) in enumerate(tqdm(dataloader,
                                                                desc=f'Epoch {epoch + 1}/{num_epochs}',
                                                                ncols=100)):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            rewards_batch = rhirl(x_batch, y_batch)

            with torch.no_grad():
                # we must ignore gradients for rsvf-esvf scale calculation
                rewards_batch_cpu = rewards_batch.detach().cpu()
                rewards_batch_cpu = rewards_batch_cpu * segment_punishments
                scales_batch = get_scales_batch(number_of_road_segments,
                                                rewards_batch_cpu,
                                                historical_trajectories,
                                                real_svf,
                                                finder_for_all_trajectories_between,
                                                time_batch)
                scales_batch = scales_batch * segment_punishments
                scales_batch = scales_batch.to(device)

            # negative because we want to optimize
            scaled_loss = -(rewards_batch * scales_batch)
            loss = scaled_loss.mean()
            loss.backward()
            optimizer.step()

            if i % 5 == 4:
                tb_writer.add_scalar('Loss', loss.item(),
                                     epoch * len(dataloader) + i)

    tb_writer.close()
    rhirl.eval()
    torch.save(rhirl.state_dict(), model_save_path)
