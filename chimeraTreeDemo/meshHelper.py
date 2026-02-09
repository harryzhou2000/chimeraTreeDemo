from discretize import SimplexMesh
import numpy as np

def get_max_edge_length(mesh: SimplexMesh):
    nodes = mesh.nodes
    cells = mesh.simplices
    assert cells.shape[1] == 3

    # Gather triangle vertices
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]

    # Edge lengths
    e01 = np.linalg.norm(p0 - p1, axis=1)
    e12 = np.linalg.norm(p1 - p2, axis=1)
    e20 = np.linalg.norm(p2 - p0, axis=1)

    max_edge_length = np.maximum.reduce([e01, e12, e20])
    return max_edge_length


def get_max_AABB_length(mesh: SimplexMesh):
    nodes = mesh.nodes
    cells = mesh.simplices
    assert cells.shape[1] == 3

    # Gather triangle vertices
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]

    pMax = np.maximum.reduce([p0, p1, p2])
    pMin = np.minimum.reduce([p0, p1, p2])
    # print(
    #     pMax - pMin,
    # )
    ret =  np.max(pMax - pMin, axis=1) 

    return ret

def get_level(h:np.ndarray, level_max:int, h_0: float):
    return np.clip(level_max - np.floor(np.log2(h / h_0)), 1, level_max)
