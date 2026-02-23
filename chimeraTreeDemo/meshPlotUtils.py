import matplotlib.pyplot as plt

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np

import discretize


def get_color_seq(alpha=0.3):
    tab10_colors = matplotlib.colormaps["tab10"].colors
    return [to_rgba(c, alpha=alpha) for c in tab10_colors]


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
    cell_line=False,
    range=None,
):
    v = cell_mask
    if v is None:
        v = np.ones(mesh0.n_cells)

    grid_opts = {}
    grid = False
    # if hasattr(mesh0, "refine_points"):
    #     grid = True
    #     grid_opts.update({"cell_line": cell_line})

    if isinstance(facecolor, str) and facecolor[0] != "#":
        cmap = facecolor
        alpha = 0.5
    else:
        cmap = monocolor_cmap(to_rgba(facecolor))
        alpha = None

    mesh0.plot_image(
        ax=ax,
        v=v,
        clim=(0, 1),
        v_type="CC",
        pcolor_opts={
            "alpha": alpha,  # this makes sure no overriding cmap's alpha
            "cmap": cmap,
            "edgecolors": "face",
            "lw": linewidth,
            "vmin": None,
            "vmax": None,
            "norm": "linear",
        },
    )
    mesh0.plot_grid(ax=ax, color=linecolor, linewidth=linewidth, **grid_opts)

    if cell_line:
        cents = np.asarray(mesh0.cell_centers).T
        ax.plot(cents[0], cents[1], "k-", linewidth=0.2)
