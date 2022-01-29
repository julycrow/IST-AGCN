import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))  # 创建一个全0数组
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))  # inward, outward是列表，列表里是（a,b）这种坐标类型 的数据
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))  # 堆叠数组
    return A


def get_spatial_sym_graph(num_node, self_link, inward, outward, sym):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))  # inward, outward是列表，列表里是（a,b）这种坐标类型 的数据
    Out = normalize_digraph(edge2mat(outward, num_node))
    Sym = normalize_digraph(edge2mat(sym, num_node))
    A = np.stack((I, In, Out, Sym))  # 堆叠数组
    return A
