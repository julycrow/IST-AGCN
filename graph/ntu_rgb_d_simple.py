import sys
import torch
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 19
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 19), (3, 19), (4, 3), (5, 19), (6, 5), (7, 6),
                    (8, 19), (9, 8), (10, 9), (11, 1), (12, 11), (13, 12),
                    (14, 13), (15, 1), (16, 15), (17, 16), (18, 17)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        if labeling_mode == 'spatial':
            self.A = self.get_adjacency_matrix(labeling_mode)
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
            return A
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
