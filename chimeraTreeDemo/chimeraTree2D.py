import meshBuilder
import numpy as np
from discretize import SimplexMesh
import distanceField
import meshHelper


class FGMesh:
    def __init__(self, mesh0: SimplexMesh, b0: np.ndarray):
        self.mesh = mesh0
        self.b = b0
        self.d = distanceField.get_mesh_nodal_wall_distance(mesh0, b0)


class ChimeraTree2D:
    def __init__(self, h0=1 / 1024, Lx=128.0, Ly=128.0, x0="CC"):
        self.meshes = []

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
        
    
