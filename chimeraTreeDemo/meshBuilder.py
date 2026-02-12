from discretize import TreeMesh, SimplexMesh
from discretize.utils import mkvc
import matplotlib.pyplot as plt
import numpy as np


class SimplexMeshShifted(SimplexMesh):
    def __init__(self, nodes, simplices):
        dim = nodes.shape[1]
        self.off = np.zeros((1, dim), dtype=np.float64)
        self.rot = np.eye(dim, dtype=np.float64)
        super().__init__(nodes, simplices)

    @classmethod
    def from_SimplexMesh(cls, mesh: SimplexMesh):
        return cls(mesh.nodes, mesh.simplices)

    @property
    def nodes(self):  # override
        return (super().nodes + self.off) @ self.rot.T

    @property
    def cell_centers(self):  # override (we could force updating after shift changing)
        return np.mean(self.nodes[self.simplices], axis=1)


def buildMesh_quadTree(dx=1.0, dy=1.0, x_len=256.0, y_len=256.0, x0="CC"):
    x_levels = int(np.ceil(np.log2(x_len / dx)))
    y_levels = int(np.ceil(np.log2(y_len / dy)))
    levels = min(x_levels, y_levels)

    # Compute number of base mesh cells required in x and y
    nbcx = 2**x_levels
    nbcy = 2**y_levels

    # Define the base mesh
    hx = [(dx, nbcx)]
    hy = [(dy, nbcy)]
    mesh = TreeMesh([hx, hy], x0=x0, diagonal_balance=False)

    return mesh, levels


#########


import gmsh


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

    x0 = np.asarray(x0, dtype=np.float64)

    # gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    gmsh.model.add("ring")
    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------
    # ci = gmsh.model.occ.addCircle(x0[0], x0[1], 0.0, r_inner)
    # co = gmsh.model.occ.addCircle(x0[0], x0[1], 0.0, r_outer)

    # li = gmsh.model.occ.addCurveLoop([-ci])
    # lo = gmsh.model.occ.addCurveLoop([co])

    # surf = gmsh.model.occ.addPlaneSurface([lo, li])

    # gmsh.model.occ.synchronize()

    # ------------------------------------------------------------
    # Geometry with GEO
    # ------------------------------------------------------------
    p_center = gmsh.model.geo.addPoint(x0[0], x0[1], 0, 0.1)
    p_inner = gmsh.model.geo.addPoint(x0[0] + r_inner, x0[1], 0, lc_inner)
    p_outer = gmsh.model.geo.addPoint(x0[0] + r_outer, x0[1], 0, lc)
    p_inner_l = gmsh.model.geo.addPoint(x0[0] - r_inner, x0[1], 0, lc_inner)
    p_outer_l = gmsh.model.geo.addPoint(x0[0] - r_outer, x0[1], 0, lc)

    ci = gmsh.model.geo.addCircleArc(p_inner, p_center, p_inner_l)  # Closed circle
    co = gmsh.model.geo.addCircleArc(p_outer, p_center, p_outer_l)

    ci_lo = gmsh.model.geo.addCircleArc(p_inner_l, p_center, p_inner)  # Closed circle
    co_lo = gmsh.model.geo.addCircleArc(p_outer_l, p_center, p_outer)

    # Curve loops (outer CCW, inner CW via negative tag)
    outer_loop = gmsh.model.geo.addCurveLoop([co, co_lo])
    inner_loop = gmsh.model.geo.addCurveLoop([-ci, -ci_lo])
    surf = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

    # Enforce uniform 1D mesh with exact point counts
    n_inner = int(np.ceil(np.pi * r_inner / lc_inner))
    n_outer = int(np.ceil(np.pi * r_outer / lc))
    gmsh.model.geo.mesh.setTransfiniteCurve(ci, n_inner, "Progression", 1.0)
    gmsh.model.geo.mesh.setTransfiniteCurve(ci_lo, n_inner, "Progression", 1.0)
    gmsh.model.geo.mesh.setTransfiniteCurve(co, n_outer, "Progression", 1.0)
    gmsh.model.geo.mesh.setTransfiniteCurve(co_lo, n_outer, "Progression", 1.0)
    gmsh.model.geo.synchronize()

    # ------------------------------------------------------------
    # Physical groups (boundary IDs)
    # ------------------------------------------------------------
    inner_tag = gmsh.model.addPhysicalGroup(1, [ci, ci_lo])
    outer_tag = gmsh.model.addPhysicalGroup(1, [co, co_lo])
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
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [ci, ci_lo])

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

    # disable some mesh size controls
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

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
