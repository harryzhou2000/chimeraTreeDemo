from discretize import TreeMesh
from discretize.utils import mkvc
import matplotlib.pyplot as plt
import numpy as np

dx = 5  # minimum cell width (base mesh cell width) in x
dy = 5  # minimum cell width (base mesh cell width) in y

x_length = 300.0  # domain width in x
y_length = 300.0  # domain width in y

# Compute number of base mesh cells required in x and y
nbcx = 2 ** int(np.ceil(np.log2(x_length / dx)))
nbcy = 2 ** int(np.ceil(np.log2(y_length / dy)))

# Define the base mesh
hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
mesh = TreeMesh([hx, hy], x0="CC")

# Refine surface topography
xx = mesh.nodes_x
yy = -3 * np.exp((xx**2) / 100**2) + 50.0
pts = np.c_[mkvc(xx), mkvc(yy)]
print(yy)
padding = [[0, 2], [0, 2]]
mesh.refine_surface(pts, padding_cells_by_level=padding, finalize=False)

# Refine mesh near points
xx = np.array([0.0, 10.0, 0.0, -10.0])
yy = np.array([-20.0, -10.0, 0.0, -10])
pts = np.c_[mkvc(xx), mkvc(yy)]
mesh.refine_points(pts, padding_cells_by_level=[2, 2], finalize=False)

mesh.finalize()

# We can apply the plot_grid method and output to a specified axes object
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
mesh.plot_grid(ax=ax)
ax.set_xbound(mesh.x0[0], mesh.x0[0] + np.sum(mesh.h[0]))
ax.set_ybound(mesh.x0[1], mesh.x0[1] + np.sum(mesh.h[1]))
ax.set_title("QuadTree Mesh")
