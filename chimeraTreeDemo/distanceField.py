from discretize import SimplexMesh

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


def get_mesh_nodal_wall_distance(mesh0: SimplexMesh, b0: np.ndarray):
    nodes = mesh0.nodes
    assert nodes.shape[1] == 2  # 2D
    pb0 = nodes[b0]

    ## using KD tree
    # pb0 = np.reshape(pb0, (-1, 2))
    # tree = KDTree(pb0)
    # distances, i = tree.query(nodes)

    # return distances

    ## using trimesh

    pb0_extend = np.zeros((pb0.shape[0], pb0.shape[1] + 1, 3), dtype=pb0.dtype)
    pb0_extend[:, 0:2, 0:2] = pb0
    pb0_extend[:, 2:3, 0:2] = pb0[:, 0:1, :]
    pb0_extend[:, 2:3, 2] = 1
    vertices = pb0_extend.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    closest, distances, triangle_ids = mesh.nearest.on_surface(
        np.concatenate([nodes, np.zeros((nodes.shape[0], 1))], axis=1)
    )

    return distances
