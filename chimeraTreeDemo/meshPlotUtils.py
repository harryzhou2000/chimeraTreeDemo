import matplotlib.pyplot as plt


from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np

import discretize


def monocolor_cmap(color=(0.5, 0.5, 0.5, 1)):
    r, g, b, a = color
    r, g, b, a = float(r), float(g), float(b), float(a)
    cdict = {
        "red": [[0.0, r, r], [1.0, r, r]],
        "green": [[0.0, g, g], [1.0, g, g]],
        "blue": [[0.0, b, b], [1.0, b, b]],
        "alpha": [
            [0.0, 0.0, 0.0],  # At data=0, alpha=0 (transparent)
            [1.0, a, a],
        ],  # At data=1, alpha=1 (opaque)
    }

    # Create the colormap
    return LinearSegmentedColormap("MonocolorAlpha", cdict)


def plot_mesh_mono(
    mesh0: discretize.SimplexMesh,
    ax,
    linewidth=0.5,
    linecolor="#0000FF7B",
    facecolor="#00EAFFB9",
    cell_mask=None,
):
    v = cell_mask
    if v is None:
        v = np.ones(mesh0.n_cells)

    mesh0.plot_image(
        ax=ax,
        v=v,
        clim=(0, 1),
        v_type="CC",
        pcolor_opts={
            "alpha": None,  # this makes sure no overriding cmap's alpha
            "cmap": monocolor_cmap(to_rgba(facecolor)),
            "edgecolors": "face",
            "lw": linewidth
        },
    )
    mesh0.plot_grid(ax=ax, color=linecolor, linewidth=linewidth)
