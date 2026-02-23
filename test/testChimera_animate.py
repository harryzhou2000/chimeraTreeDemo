from chimeraTreeDemo import chimeraTree2D
from chimeraTreeDemo import meshBuilder
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

anim = 2

def get_chimera():
    to_shift_ = meshBuilder.SimplexMeshShifted.from_SimplexMesh

    chimera = chimeraTree2D.ChimeraTree2D(h0=1 / 1024 * 1 / 2**4)
    print(chimera.n_meshes)
    if hasattr(chimera, "coons"):
        print(len(chimera.conns))

    if anim == 1:
        ## animation 1
        mesh, b = meshBuilder.buildMesh_tri_cylinder_gmsh([0, 0], 0.5, 0.8, 0.1, 0.04)
        chimera.add_tri_mesh(chimeraTree2D.FGMesh(to_shift_(mesh), b))
        mesh, b = meshBuilder.buildMesh_tri_cylinder_gmsh([2, 0], 0.5, 0.8, 0.1, 0.04)
        chimera.add_tri_mesh(chimeraTree2D.FGMesh(to_shift_(mesh), b))
    elif anim == 2:
        ## animation 2
        # mesh, b = meshBuilder.buildMesh_tri_cylinder_gmsh([0, 0], 0.5, 0.8, 0.1, 0.04)
        mesh, b = meshBuilder.buildMesh_tri_rect_gmsh(
            [0, -1], (10, 1), (10 + 2, 1 + 2.05), 0.2, 0.08
        )
        chimera.add_tri_mesh(chimeraTree2D.FGMesh(to_shift_(mesh), b))

        mesh, b = meshBuilder.buildMesh_tri_cylinder_gmsh([0, 0], 0.5, 0.8, 0.1, 0.04)
        # mesh, b = meshBuilder.buildMesh_tri_rect_gmsh([1, 0], (0.5, 0.5), (0.8, 0.8), 0.1, 0.04)

        chimera.add_tri_mesh(chimeraTree2D.FGMesh(to_shift_(mesh), b))

    # meshS = meshBuilder.SimplexMeshShifted(mesh.nodes, mesh.simplices)
    # meshS.off[:] = [1, 0]
    # chimera.add_tri_mesh(chimeraTree2D.FGMesh(meshS, b))

    return chimera


def update_and_draw(ax: Axes, chimera: chimeraTree2D.ChimeraTree2D, new_off=[0, 0]):
    
    if anim == 1:
        xc, yc = 0.5, 0
        hxsee = 2
        hysee = 1
    elif anim == 2:
        xc, yc = 0, 1
        hxsee = 3
        hysee = 2

    chimera.meshes[1].mesh.off[:] = new_off
    chimera.refine_meshB()
    chimera.build_cell_conn()
    chimera.chimeraHole(gap=1e10)
    # print(chimera.conns[0])
    from chimeraTreeDemo import meshPlotUtils

    chimera.plot_holes(ax=ax)
    # meshPlotUtils.plot_mesh_mono(
    #     chimera.meshB,
    #     ax=ax,
    #     cell_mask=chimera.fluid_solid_B == -1,
    # )
    # mesh = chimera.meshes[0]
    # mesh.mesh.plot_image(ax=ax, v=mesh.d, v_type="N")

    ax.axis("tight")
    ax.set_xlim([xc - hxsee, xc + hxsee])
    ax.set_ylim([yc - hysee, yc + hysee])
    ax.set_aspect(1, adjustable="datalim")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_axis_off()


def animate(out_base: str):
    import sys, os

    if os.path.exists(out_base):
        if not os.path.isdir(out_base):
            raise ValueError(f"out_base {out_base} invalid")
    else:
        os.makedirs(out_base)

    N_frame = 128 + 1
    if anim == 1:
        offsetXs = np.linspace(0, -1.5, N_frame)
        offsetYs = np.linspace(0, 0, N_frame)
    elif anim == 2:
        offsetXs = np.linspace(0, 0, N_frame)
        offsetYs = np.linspace(3, 0, N_frame)

    offsets = np.permute_dims(np.array([offsetXs, offsetYs]), axes=(1, 0))

    chimera = get_chimera()
    fig = plt.figure(123, figsize=(6, 4), dpi=320)
    ax = fig.add_subplot()
    for iFrame, offset in enumerate(offsets):
        ax.clear()
        update_and_draw(ax, chimera, new_off=offset)
        fig.savefig(os.path.join(out_base, f"out_{iFrame:04d}.png"))
        print(f" === Frame [{iFrame:4d}] done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("testChimera animate")
    parser.add_argument("out_path")

    args = parser.parse_args()
    animate(args.out_path)
