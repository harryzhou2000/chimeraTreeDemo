import numpy as np
from discretize import TreeMesh, SimplexMesh
import distanceField

from dataclasses import dataclass
import meshHelper


class FGMesh:
    def __init__(self, mesh0: SimplexMesh, b0: np.ndarray):
        self.mesh = mesh0
        self.b = b0
        self.d = distanceField.get_mesh_nodal_wall_distance(mesh0, b0)


@dataclass
class MeshCellConn:
    B_cells_on_M: list[np.ndarray]
    M_cells_on_B: list[np.ndarray]
    B_cells_on_Mb: list[np.ndarray]
    Mb_cells_on_B: list[np.ndarray]


def chimeraMeshBHole2DTreeTri(
    meshB: TreeMesh,
    meshes: list[FGMesh],
    meshB_cell2cell: list[np.ndarray],
    conns: list[MeshCellConn],
):
    """
    Get solid hole in meshB, -1 is solid, 2 is boundary, 1 is fluid
    """

    fluid_solid_B = np.zeros(meshB.n_cells, dtype=np.int32)
    fluid_solid_B_meshes = []

    for conn in conns:
        fluid_solid_B_mesh = np.zeros(meshB.n_cells, dtype=np.int32)
        for B_cells in conn.B_cells_on_M:
            fluid_solid_B_mesh[B_cells] = 1
        fluid_solid_B_meshes.append(fluid_solid_B_mesh)
    fluid_solid_B[np.logical_and.reduce(fluid_solid_B_meshes)] = 1  # to fluid

    fsbnd_BCells = set()
    for conn in conns:
        for B_cells in conn.B_cells_on_Mb:
            fluid_solid_B[B_cells] = 2  # to fluid_solid_boundary
            fsbnd_BCells.update(B_cells)

    solid_front = set(fsbnd_BCells)
    solid_front_new = set()
    while True:
        for iCB in solid_front:
            iCOthers = meshB_cell2cell[iCB]
            iCOthers = iCOthers[fluid_solid_B[iCOthers] == 0]
            solid_front_new.update(iCOthers)
            # for iCOther in meshB_cell2cell[iCB]:
            #     if fluid_solid_B[iCOther] == 0:
            #         solid_front_new.add(iCOther)

        solid_front_new, solid_front = solid_front, solid_front_new
        solid_front_new.clear()
        if len(solid_front) == 0:
            break
        # print(sorted(list(solid_front)))
        fluid_solid_B[np.fromiter(solid_front, dtype=np.int64)] = -1
        # print(len(solid_front))
    return fluid_solid_B


def chimeraHole2DTreeTri(
    meshB: TreeMesh,
    meshes: list[FGMesh],
    fluid_solid_B: np.typing.NDArray[np.int32],
    meshB_cell2cell: list[np.ndarray],
    conns: list[MeshCellConn],
):
    import geomHelper

    inf_val = 1e300

    holeB = np.zeros(meshB.n_cells)
    holes = [np.zeros(mesh.mesh.n_cells) for mesh in meshes]

    """
    The distance exclusive BG method
    """

    B_nodes = meshHelper.get_tree_mesh_corner_nodes(meshB)
    B_dists = np.empty((len(meshes), meshB.n_cells, 4), dtype=np.float64)
    B_dists[:] = inf_val

    for iMesh, (mesh, conn) in enumerate(zip(meshes, conns)):
        B_dist = B_dists[iMesh]
        nodes = mesh.mesh.nodes
        # tri_nodes = nodes[mesh.mesh.simplices]
        for iCellB, (M_cells, Mb_cells) in enumerate(
            zip(conn.M_cells_on_B, conn.Mb_cells_on_B)
        ):
            # M_cells_nodes = tri_nodes[M_cells] # (N_mcell, 3, 2)
            M_cells_tri = mesh.mesh.simplices[M_cells]
            # test B_nodes[iCellB] in M_cells_nodes (N_mcell triangles)
            if M_cells_tri.size:
                nodalD = geomHelper.triangle_mesh_interpolation(
                    M_cells_tri, nodes, B_nodes[iCellB], mesh.d, inf_val=inf_val
                )
            else:
                nodalD = np.empty(4, dtype=np.float64)
                nodalD[:] = inf_val
            if fluid_solid_B[iCellB] == -1:
                nodalD[:] = -inf_val
            if fluid_solid_B[iCellB] == 2:
                nodalD[nodalD > inf_val * 0.5] = -inf_val

            B_dist[iCellB] = nodalD

    B_dists_range: list[np.ndarray] = (
        B_dists.min(axis=2),
        B_dists.max(axis=2),
    )  # (2, N_mesh, N_cell)

    holeB += B_dists_range[1].min(axis=0) > 0.5 * inf_val

    holeB_overlapPos = interval_overlap_any_two(B_dists_range)
    # holeB += holeB_overlapPos

    holeB = holeB != 0

    return holeB, holes


def interval_overlap_any_two(intervals: tuple[np.ndarray, np.ndarray], gap=0.0):
    minV = intervals[0]
    maxV = intervals[1]

    n_intervals = minV.shape[0]

    overlaps = []

    for i in range(n_intervals):
        for j in range(i + 1, n_intervals):
            overlap = np.logical_not(
                np.logical_or(minV[i] - maxV[j] >= gap, minV[j] - maxV[i] >= gap)
            )
            overlaps.append(overlap)
    return np.logical_or.reduce(overlaps)
