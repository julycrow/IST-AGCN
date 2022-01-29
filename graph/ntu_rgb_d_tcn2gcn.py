import sys
import torch
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
sym_index = [(23, 25), (24, 22), (11, 7), (10, 6), (9, 5), (8, 12), (16, 20), (17, 13),
             (18, 14), (19, 15)]
sym = [(i - 1, j - 1) for (i, j) in sym_index]
inward_2 = [(0, 20), (3, 20), (5, 20), (6, 4), (7, 5), (9, 20), (10, 8),
            (11, 9), (12, 1), (13, 0), (14, 12), (12, 11), (15, 13),
            (16, 1), (17, 0), (18, 16), (19, 17), (21, 7), (22, 6),
            (23, 11), (24, 20)]
outward_2 = [(0, 13), (0, 17), (1, 2), (1, 4), (1, 8), (1, 12), (1, 14),
             (2, 1), (2, 4), (2, 8), (4, 1), (4, 2), (4, 6),
             (4, 8), (5, 7), (6, 22), (7, 21), (8, 1), (8, 2),
             (8, 4), (8, 10), (9, 11), (10, 24), (11, 23), (12, 14), (12, 16), (13, 15), (16, 12),
             (16, 18), (17, 19), (20, 0), (20, 3), (20, 5), (20, 9)]
inward_3 = [(6, 20), (7, 4), (10, 20),
            (11, 8), (12, 20), (13, 1), (14, 0), (15, 12),
            (16, 20), (17, 1), (18, 0), (19, 16), (21, 6), (22, 5),
            (23, 10), (24, 9)]
outward_3 = [(0, 2), (0, 4), (0, 8), (0, 14), (0, 18), (1, 3), (1, 5), (1, 9), (1, 13), (1, 17),
             (2, 0), (2, 5), (2, 9), (3, 1), (3, 4), (3, 8), (4, 0), (4, 3), (4, 7),
             (4, 9), (5, 1), (5, 2), (5, 8), (5, 22), (6, 21), (8, 0), (8, 3),
             (8, 5), (8, 11), (9, 1), (9, 2), (9, 4), (9, 24), (10, 23), (12, 15), (12, 17), (13, 16), (16, 13),
             (16, 19), (17, 12), (20, 6), (20, 10), (20, 12), (20, 14)]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        if labeling_mode == 'spatial' or labeling_mode == 'spatial_sym':
            self.A, self.tcn_A = self.get_adjacency_matrix(labeling_mode)
        elif labeling_mode == 'spatial_3' or labeling_mode == 'spatial_3_sym':
            self.A, self.A2, self.A3 = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
            tcn_A = torch.zeros((25, 25))

            for i in range(25):
                tcn_A[i][i] = 1
                for j in range(1, 5):
                    if i - j >= 0:
                        tcn_A[i][i - j] = 1
                for j in range(1, 5):
                    if i + j <= 24:
                        tcn_A[i][i + j] = 1
            return A, tcn_A
        elif labeling_mode == 'spatial_sym':
            A = tools.get_spatial_sym_graph(num_node, self_link, inward, outward, sym)
            tcn_A = torch.zeros((25, 25))
            for i in range(25):
                tcn_A[i][i] = 1
                for j in range(1, 5):
                    if i - j >= 0:
                        tcn_A[i][i - j] = 1
                for j in range(1, 5):
                    if i + j <= 24:
                        tcn_A[i][i + j] = 1
            return A, tcn_A
        elif labeling_mode == 'spatial_3':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
            A2 = tools.get_spatial_graph(num_node, self_link, inward_2, outward_2)
            A3 = tools.get_spatial_graph(num_node, self_link, inward_3, outward_3)
            return A, A2, A3
        elif labeling_mode == 'spatial_3_sym':
            A = tools.get_spatial_sym_graph(num_node, self_link, inward, outward, sym)
            A2 = tools.get_spatial_graph(num_node, self_link, inward_2, outward_2)
            A3 = tools.get_spatial_graph(num_node, self_link, inward_3, outward_3)
            return A, A2, A3
        else:
            raise ValueError()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial_sym').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
