"""
This file contains the proof that our trajectories violate the road network
"""


import osmnx as ox
import folium
import pandas as pd
import ast
from rhirl.model.constants import test_data_file_name


def can_go_from_to(graph, point1, point2):
    node1 = ox.distance.nearest_nodes(graph, point1[1], point1[0])
    node2 = ox.distance.nearest_nodes(graph, point2[1], point2[0])
    match_problem1 = [graph.nodes[node1]['y'],
                      graph.nodes[node1]['x']] != point1
    match_problem2 = [graph.nodes[node2]['y'],
                      graph.nodes[node2]['x']] != point2

    edge_data = graph.get_edge_data(node1, node2)
    print(edge_data)
    if match_problem1 or match_problem2:
        return "match problem"

    if graph.has_edge(node1, node2):
        return "edge exists"

    return "no edge"


if __name__ == "__main__":
    """
    Better run this on a notebook to see the map
    """
    test_df = pd.read_csv(test_data_file_name)
    test_coords_mm = [ast.literal_eval(coords_mm)
                      for coords_mm in test_df["COORDS_MM"]]

    graph = ox.graph_from_place('Porto, Portugal', network_type='drive')
    bad_route = test_coords_mm[1629][20:24]
    m = folium.Map(location=bad_route[0], zoom_start=300)

    folium.CircleMarker(location=bad_route[0], radius=10,
                        color='green', fill=True, fill_color='green').add_to(m)
    folium.CircleMarker(location=bad_route[1], radius=10,
                        color='red', fill=True, fill_color='red').add_to(m)
    folium.CircleMarker(location=bad_route[2], radius=10,
                        color='black', fill=True, fill_color='black').add_to(m)
    folium.PolyLine(bad_route, color='blue').add_to(m)

    print(bad_route)
    s, e = 1, 2
    print(can_go_from_to(graph, bad_route[s], bad_route[e]))
    print(can_go_from_to(graph, bad_route[e], bad_route[s]))
    m
