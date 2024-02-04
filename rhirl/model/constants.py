"""
This file contains flags, model paramaters and file paths
"""


import os


# Flags
"""
If set to True, the model will have two heads
and aggregate via concatenation
It will also add another dense layer to AfterGAT
"""
TWO_HEAD = True

"""
Amplify the rewards of the segments which are
consistent with OSM
"""
PUNISH = True


# Parameters for training
num_epochs = 20
batch_size = 24

"""
This is the first reduction from
n = len(adjacency_matrix) = 12956
"""
hidden_size = 2048

# Output of preGAT, hence half of the input of GAT
feature_dim = 64

"""
Doesn't need to be bigger than size of the hidden vectors in GAT
but it is suggested in https://arxiv.org/pdf/1710.10903.pdf
This is the output dim of GAT
"""
output_dim = 256

# It won't be relevant if TWO_HEAD is False
final_hidden_size = 32
lr = 0.001


# Parameters for basic speed calculation
default_speed_value = 2
minimum_speed = 1
maximum_speed = 200


# Always existing file names
raw_data_dir_name = "rhirl/data/raw"
processed_data_dir_name = "rhirl/data/processed"
train_data_file_name = os.path.join(raw_data_dir_name, 'train.csv')
val_data_file_name = os.path.join(raw_data_dir_name, 'val.csv')
test_data_file_name = os.path.join(raw_data_dir_name, 'test.csv')
saved_models_dir_name = "rhirl/training/saved_models/"
training_logs_dir_name = "rhirl/training/tb_logs/"
saved_model_params_dir_name = os.path.join(saved_models_dir_name, "weights/")
saved_model_rewards_dir_name = os.path.join(saved_models_dir_name, "rewards/")
saved_model_evals_dir_name = os.path.join(saved_models_dir_name, "evals/")


# Generated if rhirl_data_prep.py is run
train_trajectories_file_name = os.path.join(
    processed_data_dir_name, 'train_trajectories.pickle')
train_traffic_speeds_file_name = os.path.join(
    processed_data_dir_name, 'train_traffic_speeds.pt')
test_trajectories_file_name = os.path.join(
    processed_data_dir_name, 'test_trajectories.pickle')
test_traffic_speeds_file_name = os.path.join(
    processed_data_dir_name, 'test_traffic_speeds.pt')
test_times_file_name = os.path.join(
    processed_data_dir_name, 'test_times.npy')
train_finder_for_all_trajectories_between_file_name = os.path.join(
    processed_data_dir_name, 'train_finder_for_all_trajectories_between.pickle')
integer_segment_to_coords_file_name = os.path.join(
    processed_data_dir_name, 'integer_segment_to_coords.pickle')
historical_trajectories_file_name = os.path.join(
    processed_data_dir_name, 'historical_trajectories.pickle')
adjacency_matrix_file_name = os.path.join(
    processed_data_dir_name, 'adjacency_matrix.npy')
segment_punishments_file_name = os.path.join(
    processed_data_dir_name, 'segment_punishments.npy')
real_svf_file_name = os.path.join(processed_data_dir_name, 'real_svf.pt')
