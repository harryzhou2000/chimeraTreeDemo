import numpy as np


def triangle_mesh_interpolation(tri, coords, pts, d, tol=1e-12, inf_val=1e300):
    """
    Fully vectorized 2D triangle mesh interpolation.

    Parameters
    ----------
    tri : ndarray, shape (N, 3)
        Triangle connectivity (node indices for each triangle).
    coords : ndarray, shape (N_node, 2)
        Node coordinates (x, y).
    pts : ndarray, shape (N_pt, 2)
        Query points to interpolate.
    d : ndarray, shape (N_node,)
        Nodal data values.
    tol : float, optional
        Numerical tolerance for inside-triangle test (default: 1e-12).

    Returns
    -------
    result : ndarray, shape (N_pt,)
        Interpolated values at query points. Points outside all triangles
        are assigned np.inf.
    """
    N, N_pt = tri.shape[0], pts.shape[0]
    assert N > 0 and N_pt > 0

    # Get vertex coordinates for all triangles: (N, 3, 2)
    tri_coords = coords[tri]

    # Broadcasting preparation:
    # tri_coords_bc: (N, 1, 3, 2), pts_bc: (1, N_pt, 1, 2)
    tri_coords_bc = tri_coords[:, np.newaxis, :, :]
    pts_bc = pts[np.newaxis, :, np.newaxis, :]

    # Extract vertices with automatic broadcasting -> (N, N_pt, 2)
    v0 = tri_coords_bc[:, :, 0, :]
    v1 = tri_coords_bc[:, :, 1, :]
    v2 = tri_coords_bc[:, :, 2, :]
    p = pts_bc[:, :, 0, :]  # Broadcasts to (N, N_pt, 2)

    # Compute denominator = 2 * signed area of triangle (v0, v1, v2)
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    denom = v0v1[..., 0] * v0v2[..., 1] - v0v1[..., 1] * v0v2[..., 0]  # (N, N_pt)

    # Compute barycentric numerators (2 * signed sub-areas)
    # w0 ∝ area(p, v1, v2)
    v1p = v1 - p
    v2p = v2 - p
    area0 = v1p[..., 0] * v2p[..., 1] - v1p[..., 1] * v2p[..., 0]

    # w1 ∝ area(p, v2, v0)
    v2p = v2 - p
    v0p = v0 - p
    area1 = v2p[..., 0] * v0p[..., 1] - v2p[..., 1] * v0p[..., 0]

    # w2 ∝ area(p, v0, v1)
    v0p = v0 - p
    v1p = v1 - p
    area2 = v0p[..., 0] * v1p[..., 1] - v0p[..., 1] * v1p[..., 0]

    # Safe division: mask degenerate triangles (|denom| < tol)
    valid_tri = np.abs(denom) >= tol
    denom_safe = np.where(valid_tri, denom, 1.0)  # Avoid division by zero

    # Barycentric coordinates
    w0 = np.where(valid_tri, area0 / denom_safe, -1.0)
    w1 = np.where(valid_tri, area1 / denom_safe, -1.0)
    w2 = np.where(valid_tri, area2 / denom_safe, -1.0)

    # Inside test: all weights >= -tol (handles both CCW/CW orientations)
    inside = (w0 >= -tol) & (w1 >= -tol) & (w2 >= -tol) & valid_tri  # (N, N_pt)

    # For each point, find first containing triangle (or 0 if none)
    triangle_idx = np.argmax(inside, axis=0)  # (N_pt,)
    point_valid = np.any(inside, axis=0)  # (N_pt,)

    # Extract barycentric weights for selected triangles
    w0_sel = w0[triangle_idx, np.arange(N_pt)]
    w1_sel = w1[triangle_idx, np.arange(N_pt)]
    w2_sel = w2[triangle_idx, np.arange(N_pt)]

    # Get nodal data for vertices of selected triangles
    tri_sel = tri[triangle_idx]  # (N_pt, 3)
    d_vals = d[tri_sel]  # (N_pt, 3)

    # Interpolate using barycentric weights
    interpolated = w0_sel * d_vals[:, 0] + w1_sel * d_vals[:, 1] + w2_sel * d_vals[:, 2]

    # Assign np.inf to points outside all triangles
    result = np.where(point_valid, interpolated, inf_val)

    return result
