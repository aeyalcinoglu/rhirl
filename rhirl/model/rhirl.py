"""
This file contains the implementation
of the reward generating model
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PreGAT(nn.Module):
    """
    From start to concatenation before GAT
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(PreGAT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.act = nn.functional.elu

        self.dense1_segment = nn.Linear(input_size, hidden_size)
        self.batchnorm1_segment = nn.BatchNorm1d(input_size * hidden_size)
        self.dense2_segment = nn.Linear(hidden_size, output_size)
        self.batchnorm2_segment = nn.BatchNorm1d(input_size * output_size)

        self.dense1_speed = nn.Linear(input_size, hidden_size)
        self.batchnorm1_speed = nn.BatchNorm1d(hidden_size)
        self.dense2_speed = nn.Linear(hidden_size, output_size)
        self.batchnorm2_speed = nn.BatchNorm1d(output_size)

    def forward(self, x, y):
        batch_size = x.size(0)

        x = self.dense1_segment(x)
        # have to merge all scenes for batchnorm
        x = x.view(batch_size, -1)
        x = self.batchnorm1_segment(x)
        x = x.view(batch_size, self.input_size, self.hidden_size)

        x = self.act(x)
        x = self.dense2_segment(x)

        x = x.view(batch_size, -1)
        x = self.batchnorm2_segment(x)
        x = x.view(batch_size, self.input_size, self.output_size)
        x = self.act(x)

        y = self.dense1_speed(y)
        y = self.batchnorm1_speed(y)
        y = self.act(y)
        y = self.dense2_speed(y)
        y = self.batchnorm2_speed(y)
        y = self.act(y)

        # traffic speed is fixed during a scene
        # it is repeated for each road segment at the end
        # instead of getting passed every single time
        y_expanded = y.unsqueeze(1).repeat(1, self.input_size, 1)
        combined = torch.cat((x, y_expanded), dim=2)

        return combined


class GAT(nn.Module):
    def __init__(self, feature_dim, output_dim, adjacency_matrix):
        super(GAT, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.feature_dim = feature_dim
        self.W_1 = nn.Parameter(torch.zeros(size=(feature_dim, feature_dim)))
        self.W_2 = nn.Parameter(torch.zeros(size=(feature_dim, output_dim)))

        # this '2*' doesn't have anything to do with head count
        self.a = nn.Parameter(torch.zeros(size=(2*feature_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.node_count = adjacency_matrix.shape[0]

        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h):
        assert h.size(1) == self.node_count
        device = h.device

        g = torch.matmul(h, self.W_1)
        edges = self.adjacency_matrix.nonzero(as_tuple=True)
        g_i = g[:, edges[0]]
        g_j = g[:, edges[1]]

        e_ij_input = torch.cat([g_i, g_j], dim=-1)
        e_ij = self.leakyrelu(torch.matmul(e_ij_input, self.a)).squeeze(-1)

        final_attention = torch.zeros_like(e_ij, device=device)
        for node in range(self.node_count):
            node_edges = (edges[0] == node)
            if not node_edges.any():
                continue
            node_e_ij = e_ij[:, node_edges]
            node_attention = nn.functional.softmax(node_e_ij, dim=1)
            final_attention[:, node_edges] = node_attention

        g_j_weighted = g_j * final_attention.unsqueeze(-1)
        h_prime = torch.zeros_like(g, device=device)
        h_prime.index_add_(1, edges[0], g_j_weighted)
        h_prime = torch.matmul(h_prime, self.W_2)

        return nn.functional.relu(h_prime)


class AfterGAT(nn.Module):
    def __init__(self, input_dim, two_head=False, final_hidden_size=None):
        super(AfterGAT, self).__init__()
        self.two_head = two_head
        num_heads = 2 if two_head else None
        if two_head:
            self.dense1 = nn.Linear(input_dim * num_heads, final_hidden_size)
            self.dense2 = nn.Linear(final_hidden_size, 1)
        else:
            self.dense = nn.Linear(input_dim, 1)

    def forward(self, h_prime):
        if self.two_head:
            h_prime = self.dense1(h_prime)
            h_prime = nn.functional.relu(h_prime)
            h_prime = self.dense2(h_prime)
            h_prime = torch.nn.functional.softsign(h_prime) - 1
        else:
            h_prime = self.dense(h_prime)
            h_prime = torch.nn.functional.softsign(h_prime) - 1

        return h_prime


class RHIRL(nn.Module):
    def __init__(self, input_dim, hidden_size,
                 feature_dim, output_dim,
                 adjacency_matrix, two_head=False, final_hidden_size=1):
        super(RHIRL, self).__init__()
        self.two_head = two_head
        self.pre_gat = PreGAT(input_dim, hidden_size, feature_dim)
        self.gat = GAT(2 * feature_dim,
                       output_dim,
                       adjacency_matrix)
        if two_head:
            self.gat_two = GAT(2 * feature_dim,
                               output_dim,
                               adjacency_matrix)
        self.after_gat = AfterGAT(output_dim, two_head, final_hidden_size)

    def forward(self, x, y):
        h = self.pre_gat(x, y)
        h_prime = self.gat(h)
        if self.two_head:
            h_prime_two = self.gat_two(h)
            reward = self.after_gat(torch.cat([h_prime, h_prime_two], dim=-1))
        else:
            reward = self.after_gat(h_prime)
        return reward


class RHIRLDataset(Dataset):
    """
    x is one-hot encoded for a single scene
    x has dimension (number of road segments) x (number of road segments)
    y is traffic speed vector for a single scene
    y has dimension (number of road segments)
    """

    def __init__(self, traffic_speeds):
        self.traffic_speeds = traffic_speeds

    def __len__(self):
        return len(self.traffic_speeds)

    def __getitem__(self, idx):
        time = list(self.traffic_speeds.keys())[idx]
        y_scene = self.traffic_speeds[time]
        x_scene = torch.eye(len(y_scene))
        return x_scene, y_scene, time
