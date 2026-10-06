import numpy as np


def read_msd_dat(path):
    """fix ave/time vector file -> steps[n], msd[n, n_mito]."""
    steps, blocks, lines = [], [], [l for l in open(path) if not l.startswith("#")]
    i = 0
    while i < len(lines):
        step, nrows = map(int, lines[i].split())
        steps.append(step)
        blocks.append([float(l.split()[1]) for l in lines[i + 1:i + 1 + nrows]])
        i += 1 + nrows
    return np.array(steps), np.array(blocks)


def read_dump_com(path):
    """Dump with id mol type and xu yu zu (or x y z ix iy iz) -> steps, COM[n_frames, n_mito, 3]."""
    steps, coms, box = [], [], None
    with open(path) as f:
        while f.readline().startswith("ITEM: TIMESTEP"):
            steps.append(int(f.readline()))
            f.readline(); n = int(f.readline())
            f.readline()
            box = np.array([np.ptp([float(v) for v in f.readline().split()[:2]]) for _ in range(3)])
            cols = f.readline().split()[2:]
            data = np.loadtxt([f.readline() for _ in range(n)])
            c = {name: k for k, name in enumerate(cols)}
            if "xu" in c:
                pos = data[:, [c["xu"], c["yu"], c["zu"]]]
            else:  # unwrap with image flags
                pos = data[:, [c["x"], c["y"], c["z"]]] + data[:, [c["ix"], c["iy"], c["iz"]]] * box
            mol = data[:, c["mol"]].astype(int)
            ids = np.unique(mol[mol > 0])
            coms.append([pos[mol == m].mean(axis=0) for m in ids])
    return np.array(steps), np.array(coms)


def msd_time_averaged(com):
    """MSD(lag) averaged over all time origins and all mitochondria -> lag frames, msd, per-mito msd."""
    n = len(com)
    lags = np.arange(1, n)
    per_mito = np.array([((com[l:] - com[:-l]) ** 2).sum(axis=2).mean(axis=0) for l in lags])
    return lags, per_mito.mean(axis=1), per_mito


def fit_power_law(t, msd):
    m = (t > 0) & (msd > 0)
    alpha, logc = np.polyfit(np.log(t[m]), np.log(msd[m]), 1)
    return alpha, np.exp(logc)