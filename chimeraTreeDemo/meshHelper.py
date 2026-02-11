from discretize import SimplexMesh, TreeMesh
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
    ret = np.max(pMax - pMin, axis=1)

    return ret


def get_level(h: np.ndarray, level_max: int, h_0: float):
    return np.clip(level_max - np.floor(np.log2(h / h_0)), 1, level_max)


def refine_tree_mesh_by_tri(
    meshB: TreeMesh, h0: float, nLevel: int, mesh0: SimplexMesh
):
    cents = mesh0.cell_centers
    Ls = get_max_edge_length(mesh0)
    LsAABB = get_max_AABB_length(mesh0)
    meshB.refine_ball(
        cents,
        Ls * 1,
        get_level(LsAABB, nLevel, h0),
        finalize=False,
    )

def get_tri_to_tree_intersect(
    meshB:TreeMesh, mesh0:SimplexMesh
):
    cells = mesh0.simplices
    nodes = mesh0.nodes

    tris = nodes[cells]

    nCell = mesh0.n_cells

    tree_cells_on_tri = [[] for _ in range(nCell)]
    tri_cells_on_tree = [[] for _ in range(meshB.n_cells)]

    for i in range(nCell):
        tree_cells_on_tri[i] = meshB.get_cells_in_triangle(tris[i])
        for iTree in tree_cells_on_tri[i]:
            tri_cells_on_tree[iTree].append(i)

    tri_cells_on_tree = [np.unique(np.array(row, dtype=np.int64)) for row in tri_cells_on_tree]

    return tree_cells_on_tri, tri_cells_on_tree
