import numpy as np

def capsule(center, axis, R=1.0, L=8.0, s=0.5):
    """Bead positions filling a capsule (spherocylinder) on a grid of spacing s."""
    g = np.arange(-L / 2, L / 2 + s, s)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    p = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    zc = np.clip(p[:, 2], -(L / 2 - R), L / 2 - R)
    p = p[p[:, 0] ** 2 + p[:, 1] ** 2 + (p[:, 2] - zc) ** 2 <= R ** 2]
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    v, c = np.cross([0, 0, 1], a), a[2]
    if np.linalg.norm(v) > 1e-9:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = p @ (np.eye(3) + K + K @ K / (1 + c)).T
    return p + center


def dist_to_segment(p, c, a, half, BOX):
    """Distance from points p to the axis segment of a capsule (periodic box)."""
    d = p - c
    d -= BOX * np.round(d / BOX)                  # minimum image
    t = np.clip(d @ a, -half, half)
    return np.linalg.norm(d - t[:, None] * a, axis=1)

def sphere(center, R=1.0, s=0.5):
    """Bead positions filling a sphere on a grid of spacing s."""
    g = np.arange(-R, R + s, s)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    p = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    p = p[np.linalg.norm(p, axis=1) <= R]
    return p + center

def dist_to_point(p, c, BOX):
    """Distance from points p to a point c (periodic box)."""
    d = p - c
    d -= BOX * np.round(d / BOX)                  # minimum image
    return np.linalg.norm(d, axis=1)

def shell(center, R=1.0, s=0.5):
    """Bead positions on the surface of a sphere on a grid of spacing s."""
    g = np.arange(-R, R + s, s)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    p = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    p = p[np.abs(np.linalg.norm(p, axis=1) - R) <= 0.5 * s]
    return p + center

def shell_bonds(p, s):
    """Bonds between shell beads closer than 1.5 s -> i, j (local indices), lengths."""
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    i, j = np.nonzero(np.triu((d > 0) & (d < 1.5 * s)))
    return i, j, d[i, j]
