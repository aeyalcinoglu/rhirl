"""
This file tests the Dijkstra baseline model
"""


import torch
from rhirl.model.graph_play import get_directed_line_matrix
from rhirl.evaluation.rhirl_baselines import Dijkstra


def test_dijkstra():
    # This is the minimal graph such that
    # one can test Dijkstra on line graph
    original_adj_matrix = torch.tensor([[0, 1, 0, 0, 0],
                                        [0, 0, 1, 1, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0]], dtype=torch.float)

    unique_edges, line_adj_matrix = get_directed_line_matrix(
        original_adj_matrix.numpy())
    line_adj_matrix = torch.tensor(line_adj_matrix, dtype=torch.float)
    reversed_edge_index = {index: edge for index,
                           edge in enumerate(unique_edges)}

    start, end = 0, 2
    for weight_add in range(2):
        segment_index_to_distance = [1000, 1, 1000, 1, 2+weight_add]
        shortest_path = Dijkstra(
            [[start, end]], line_adj_matrix, segment_index_to_distance)[0]
        clean_shortest_path = []
        for segment in shortest_path:
            clean_shortest_path.append(reversed_edge_index[segment])

        print(clean_shortest_path)


if __name__ == "__main__":
    test_dijkstra()
