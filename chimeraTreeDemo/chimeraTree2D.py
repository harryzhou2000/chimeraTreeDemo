import meshBuilder
import numpy as np
from discretize import SimplexMesh
import distanceField
import meshHelper
import chimeraTree2D_Helper
from chimeraTree2D_Helper import FGMesh, MeshCellConn


class ChimeraTree2D:
    meshes: list[FGMesh] = []

    def __init__(self, h0=1 / 1024, Lx=128.0, Ly=128.0, x0="CC"):
        self.h0 = h0
        self.meshB, self.nlevel = meshBuilder.buildMesh_quadTree(
            h0,
            h0,
            Lx,
            Ly,
            x0=x0,
        )

    def add_tri_mesh(self, mesh0: FGMesh):
        self.meshes.append(mesh0)

    def refine_meshB(self):
        for m in self.meshes:
            meshHelper.refine_tree_mesh_by_tri(
                self.meshB,
                self.h0,
                self.nlevel,
                m.mesh,
            )
        self.meshB.finalize()

    def build_cell_conn(self):
        self.meshB_cell2cell = meshHelper.get_tree_adj_face(self.meshB, self.h0)
        self.conns = [
            MeshCellConn(
                *(
                    meshHelper.get_tri_to_tree_intersect(self.meshB, mesh0.mesh)
                    + meshHelper.get_tribnd_to_tree_intersect(
                        self.meshB, mesh0.mesh, mesh0.b
                    )
                )
            )
            for mesh0 in self.meshes
        ]

    def chimeraHole(self):
        self.fluid_solid_B = chimeraTree2D_Helper.chimeraMeshBHole2DTreeTri(
            self.meshB, self.meshes, self.meshB_cell2cell, self.conns
        )
        self.holeB, self.holes = chimeraTree2D_Helper.chimeraHole2DTreeTri(
            self.meshB, self.meshes, self.fluid_solid_B, self.meshB_cell2cell, self.conns
        )
        
        
