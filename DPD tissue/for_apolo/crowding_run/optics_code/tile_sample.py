import numpy as np

DIAM_UM = 1000.0      # sample diameter (1 mm)
RC_UM = 0.25          # one DPD length unit in um
BOX_RC = 24.0         # tile = one simulation box (in r_c)
FRAME_S = 5.0         # time between saved frames (= one hologram)
RECORD_FRAMES = 1400  # holograms in the recording
EQUIL_FRAMES = 60     # frames thrown away at the start (300 s)
DELAY_FRAMES = 20     # spacing between delay slots (100 s)
N_DELAYS = 20

# the 8 symmetries of a square in the xy plane (z untouched)
ORIENT = [np.array(m) for m in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
                                [[-1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]])]


def build_layout(n_runs, diam_um=DIAM_UM, seed=0):
    """Square grid of tiles whose centres fall inside the circle; each tile gets a run, a delay slot
    and an orientation. Neighbouring tiles never share the same (run, delay)."""
    rng = np.random.default_rng(seed)
    L = BOX_RC * RC_UM
    n = int(np.ceil(diam_um / L / 2))
    ij = np.array([(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)])
    ij = ij[np.hypot(*(ij.T * L)) <= diam_um / 2]
    run = rng.integers(n_runs, size=len(ij))
    delay = rng.integers(N_DELAYS, size=len(ij))
    index = {tuple(p): k for k, p in enumerate(ij)}
    for k, (i, j) in enumerate(ij):                      # re-draw clashes with already-assigned neighbours
        nb = [index[(i + a, j + b)] for a in (-1, 0, 1) for b in (-1, 0, 1)
              if (a or b) and (i + a, j + b) in index and index[(i + a, j + b)] < k]
        while any(run[m] == run[k] and delay[m] == delay[k] for m in nb):
            run[k], delay[k] = rng.integers(n_runs), rng.integers(N_DELAYS)
    return dict(centre_um=ij * L, run=run, delay=delay, orient=rng.integers(8, size=len(ij)))


def frame_positions(layout, trajs, k, part="com", tiles=None, stride=1):
    """Positions (um) of every organelle ("com") or membrane bead ("mem") at recording frame k.
    tiles: indices of the tiles to include (default all). stride: keep every n-th membrane bead."""
    tiles = np.arange(len(layout["run"])) if tiles is None else tiles
    out = []
    for t in tiles:
        f = EQUIL_FRAMES + layout["delay"][t] * DELAY_FRAMES + k
        p = trajs[layout["run"][t]][part][f]
        if part == "mem":
            p = p[::stride]
        xy = p[:, :2] @ ORIENT[layout["orient"][t]].T
        out.append(np.c_[xy * RC_UM + layout["centre_um"][t], p[:, 2] * RC_UM])
    return np.vstack(out)


def tiles_in_square(layout, x0, y0, width):
    """Indices of tiles whose centre lies in the square [x0, x0+width) x [y0, y0+width) (um)."""
    c = layout["centre_um"]
    return np.nonzero((c[:, 0] >= x0) & (c[:, 0] < x0 + width) & (c[:, 1] >= y0) & (c[:, 1] < y0 + width))[0]


if __name__ == "__main__":
    trajs = [np.load(f"runs/cell_{k}/traj.npz") for k in range(4)]
    need = EQUIL_FRAMES + (N_DELAYS - 1) * DELAY_FRAMES + RECORD_FRAMES
    have = len(trajs[0]["steps"])
    if have < need:
        print(f"warning: runs have {have} frames, the layout needs {need}")
    layout = build_layout(len(trajs))
    np.savez("layout.npz", **layout)
    print(len(layout["run"]), "tiles of", BOX_RC * RC_UM, "um across a", DIAM_UM, "um circle")