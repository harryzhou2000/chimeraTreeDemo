from . import distanceField
from . import meshHelper
from .meshBuilder import SimplexMeshShifted

import numpy as np
from discretize import TreeMesh, SimplexMesh
from dataclasses import dataclass
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.collections


class FGMesh:
    def __init__(self, mesh0: SimplexMesh | SimplexMeshShifted, b0: np.ndarray):
        self.mesh = mesh0
        self.b = b0
        self.d = distanceField.get_mesh_nodal_wall_distance(mesh0, b0)

    def plot_mesh_mono(
        self,
        ax: Axes,
        linewidth=0.5,
        linecolor="#0000FF7B",
        facecolor="#00EAFFB9",
        cell_mask=None,
    ):
        from . import meshPlotUtils

        meshPlotUtils.plot_mesh_mono(
            self.mesh,
            ax=ax,
            linewidth=linewidth,
            linecolor=linecolor,
            facecolor=facecolor,
            cell_mask=cell_mask,
        )

    def plot_bnd(self, ax: Axes, linewidth=0.5, linecolor="#000000F6"):
        line_col = matplotlib.collections.LineCollection(
            self.mesh.nodes[self.b], colors=linecolor, linewidths=linewidth
        )
        ax.add_collection(line_col)


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

    fluid_solid_B_meshes = np.zeros((len(conns), meshB.n_cells), dtype=np.int32)

    for iMesh, conn in enumerate(conns):
        fluid_solid_B_mesh = fluid_solid_B_meshes[iMesh]
        for B_cells in conn.B_cells_on_M:
            fluid_solid_B_mesh[B_cells] = 1

        fsbnd_BCells = set()
        for B_cells in conn.B_cells_on_Mb:
            fluid_solid_B_mesh[B_cells] = 2  # to fluid_solid_boundary
            fsbnd_BCells.update(B_cells)

        solid_front = set(fsbnd_BCells)
        solid_front_new = set()
        while True:
            for iCB in solid_front:
                iCOthers = meshB_cell2cell[iCB]
                iCOthers = iCOthers[fluid_solid_B_mesh[iCOthers] == 0]
                solid_front_new.update(iCOthers)

            solid_front_new, solid_front = solid_front, solid_front_new
            solid_front_new.clear()
            if len(solid_front) == 0:
                break
            # print(sorted(list(solid_front)))
            fluid_solid_B_mesh[np.fromiter(solid_front, dtype=np.int64)] = -1
            # print(len(solid_front))

    fluid_solid_B = np.zeros(meshB.n_cells, dtype=np.int32)

    # fluid
    fluid_solid_B[np.logical_or.reduce(fluid_solid_B_meshes == 1, axis=0)] = 1
    # f-s bnd
    fluid_solid_B[np.logical_or.reduce(fluid_solid_B_meshes == 2, axis=0)] = 2
    # solid
    fluid_solid_B[np.logical_or.reduce(fluid_solid_B_meshes == -1, axis=0)] = -1

    return fluid_solid_B, fluid_solid_B_meshes


def chimeraHole2DTreeTri(
    meshB: TreeMesh,
    meshes: list[FGMesh],
    fluid_solid_B: np.typing.NDArray[np.int32],
    fluid_solid_B_meshes: np.typing.NDArray[np.int32],
    meshB_cell2cell: list[np.ndarray],
    conns: list[MeshCellConn],
    gap=0.0,
):
    from . import geomHelper

    inf_val = 1e300

    holeB = np.zeros(meshB.n_cells)
    holes = [np.zeros(mesh.mesh.n_cells) for mesh in meshes]

    """
    The distance exclusive BG method
    """

    B_nodes = meshHelper.get_tree_mesh_corner_nodes(meshB)
    B_dists = np.empty((len(meshes), meshB.n_cells, 4), dtype=np.float64)
    B_dists[:] = inf_val

    # 1. get B_dists, distance field on bg mesh

    for iMesh, (mesh, conn) in enumerate(zip(meshes, conns)):
        fluid_solid_B_mesh = fluid_solid_B_meshes[iMesh]
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
            if fluid_solid_B_mesh[iCellB] == -1:
                nodalD[:] = -inf_val
            if fluid_solid_B_mesh[iCellB] == 2:
                nodalD[nodalD > inf_val * 0.5] = -inf_val

            B_dist[iCellB] = nodalD

    # print("see 1951")
    # print(fluid_solid_B[1951])
    # print(B_dists[0][1951])

    B_dists_range: tuple[np.ndarray] = (
        B_dists.min(axis=2),
        B_dists.max(axis=2),
    )  # (2, N_mesh, N_cell)

    minDs, maxDs = B_dists_range

    # print(maxDs[0][1951])

    nMesh = len(meshes)

    B_mesh_dHoles = []

    for iMesh in range(nMesh):
        where_mask = np.arange(nMesh).reshape(nMesh, 1) != iMesh
        min_other = np.min(B_dists_range[0], axis=0, where=where_mask, initial=inf_val)
        max_other = np.max(B_dists_range[1], axis=0, where=where_mask, initial=-inf_val)
        B_mesh_dHole = maxDs[iMesh] - min_other < -gap
        B_mesh_dHoles.append(B_mesh_dHole)

    for iMesh, conn in enumerate(conns):
        for iCellB in np.arange(meshB.n_cells)[B_mesh_dHoles[iMesh]]:
            holes[iMesh][conn.M_cells_on_B[iCellB]] = 1
    holeB_overlapPos = interval_overlap_any_two(B_dists_range)
    for iMesh, conn in enumerate(conns):
        for iCellB in np.arange(meshB.n_cells)[holeB_overlapPos]:
            holes[iMesh][conn.M_cells_on_B[iCellB]] = 0

    holeB_mends = []
    # holeB_mend[iMesh] denotes the B-cells not fully covered by any active M-cell
    # except for solid and s-f bound cells
    for iMesh, (mesh, conn) in enumerate(zip(meshes, conns)):
        holeB_mend = np.zeros(meshB.n_cells, dtype=np.bool_)
        for iCellM in np.arange(mesh.mesh.n_cells)[holes[iMesh] == 0]:
            holeB_mend[conn.B_cells_on_M[iCellM]] = np.True_
        # touched by hole-fluid M-mesh cell, or not touched by any M-mesh cell at all (inf)
        holeB_mend = np.logical_or(holeB_mend, maxDs[iMesh] > 0.5 * inf_val)
        holeB_mends.append(holeB_mend)
    holeB += np.logical_and.reduce(holeB_mends)

    # # far field B-cells, needed for original distance-overlapping hole
    # holeB += B_dists_range[1].min(axis=0) > 0.5 * inf_val

    holeB = holeB != 0

    # holeB = holeB_overlapPos
    # holeB = B_mesh_dHoles[0]

    return holeB, holes


def interval_overlap_any_two(
    intervals: tuple[np.ndarray, np.ndarray], gap=0.00, inf_val=1e300
):
    minV = intervals[0]
    maxV = intervals[1]

    n_intervals = minV.shape[0]

    overlaps = []

    for i in range(n_intervals):
        for j in range(i + 1, n_intervals):
            overlap = np.logical_not(
                np.logical_or(
                    minV[i] - maxV[j] > gap,
                    minV[j] - maxV[i] > gap,
                )
            )
            overlaps.append(overlap)
    return np.logical_and(
        np.logical_or.reduce(overlaps),
        np.logical_not(
            np.logical_or.reduce(
                np.logical_or(minV > inf_val * 0.5, maxV < -inf_val * 0.5), axis=0
            )
        ),
    )
    # return np.logical_or.reduce(overlaps)
