from discretize import TreeMesh
from discretize.utils import mkvc
import matplotlib.pyplot as plt
import numpy as np


def buildMesh_quadTree(dx=1, dy=1, x_len=256, y_len=256, x0="CC"):
    x_levels = int(np.ceil(np.log2(x_len / dx)))
    y_levels = int(np.ceil(np.log2(y_len / dy)))
    levels = min(x_levels, y_levels)

    # Compute number of base mesh cells required in x and y
    nbcx = 2**x_levels
    nbcy = 2**y_levels

    # Define the base mesh
    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    mesh = TreeMesh([hx, hy], x0="CC")

    return mesh, levels


#########


import gmsh
import numpy as np
from discretize import SimplexMesh


def extract_boundary_lines(physical_name, tag_to_idx):
    dim = 1
    tag = gmsh.model.getPhysicalGroups(dim)
    phys_tag = None

    for d, t in tag:
        if gmsh.model.getPhysicalName(d, t) == physical_name:
            phys_tag = t
            break

    if phys_tag is None:
        raise RuntimeError(f"Physical group '{physical_name}' not found")

    lines = []

    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)

    for ent in entities:
        etypes, _, enodes = gmsh.model.mesh.getElements(dim, ent)
        for etype, nodes in zip(etypes, enodes):
            name = gmsh.model.mesh.getElementProperties(etype)[0]
            if name != "Line 2":
                continue

            conn = np.array(nodes).reshape(-1, 2)
            lines.append(np.vectorize(tag_to_idx.get)(conn))

    return np.vstack(lines)


def buildMesh_tri_cylinder_gmsh(
    x0,
    r_inner,
    r_outer,
    lc,
    lc_inner,
):
    """
    Build a 2-D triangular ring mesh using gmsh and return a discretize SimplexMesh.

    Parameters
    ----------
    x0 : array-like (2,)
        Center of the ring.
    r_inner : float
        Inner radius.
    r_outer : float
        Outer radius.
    lc : float
        Target mesh size.

    Returns
    -------
    mesh : discretize.SimplexMesh
    boundary_tags : dict
        Physical tags for inner / outer boundaries.
    """

    gmsh.initialize()
    gmsh.model.add("ring")

    x0 = np.asarray(x0, dtype=float)

    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------
    ci = gmsh.model.occ.addCircle(x0[0], x0[1], 0.0, r_inner)
    co = gmsh.model.occ.addCircle(x0[0], x0[1], 0.0, r_outer)

    li = gmsh.model.occ.addCurveLoop([ci])
    lo = gmsh.model.occ.addCurveLoop([co])

    surf = gmsh.model.occ.addPlaneSurface([lo, li])

    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------
    # Physical groups (boundary IDs)
    # ------------------------------------------------------------
    inner_tag = gmsh.model.addPhysicalGroup(1, [ci])
    outer_tag = gmsh.model.addPhysicalGroup(1, [co])
    surface_tag = gmsh.model.addPhysicalGroup(2, [surf])

    gmsh.model.setPhysicalName(1, inner_tag, "inner")
    gmsh.model.setPhysicalName(1, outer_tag, "outer")
    gmsh.model.setPhysicalName(2, surface_tag, "domain")

    # ------------------------------------------------------------
    # Mesh controls
    # ------------------------------------------------------------
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_inner)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    # ------------------------------------------------------------
    # Distance field from inner boundary
    # ------------------------------------------------------------
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [ci])

    # ------------------------------------------------------------
    # Threshold field: size vs distance
    # ------------------------------------------------------------
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_inner)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r_outer - r_inner)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)

    # ------------------------------------------------------------
    # Extract nodes
    # ------------------------------------------------------------
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = node_coords.reshape(-1, 3)[:, :2]

    # Map gmsh node tags → zero-based indexing
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

    # ------------------------------------------------------------
    # Extract triangles
    # ------------------------------------------------------------
    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2)

    triangles = []
    for etype, nodes in zip(elem_types, elem_nodes):
        print(gmsh.model.mesh.getElementProperties(etype)[0])
        if gmsh.model.mesh.getElementProperties(etype)[0] == "Triangle 3":
            conn = np.array(nodes).reshape(-1, 3)
            triangles.append(np.vectorize(tag_to_idx.get)(conn))

    simplices = np.vstack(triangles)

    wall_lines = extract_boundary_lines("inner", tag_to_idx)

    gmsh.finalize()

    mesh = SimplexMesh(points, simplices)

    boundary_tags = {
        "inner": inner_tag,
        "outer": outer_tag,
        "surface": surface_tag,
    }

    return mesh, wall_lines


if __name__ == "__main__":
    m, b = buildMesh_tri_cylinder_gmsh(
        [0, 0],
        1,
        2,
        0.1,
        0.05,
    )
    print(np.linalg.norm(m.nodes[b], axis=2))
