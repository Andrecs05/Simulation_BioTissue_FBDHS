import numpy as np
from .geometry import capsule, dist_to_segment, sphere, dist_to_point
from pathlib import Path



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

class Type:
    """Parameter set for one kind of object (e.g. mitochondria)."""
    def __init__(self, name, n, geo, r, l=None, s=0.5):
        self.name, self.n, self.geo, self.r, self.l, self.s = name, n, geo, r, l, s
        self.positions = []          # sphere: c ; capsule: (c, a)
        self.atom_type = None        # set by Run.add_type

    def beads(self, pos):
        if self.geo == "sphere":
            return sphere(pos, self.r, self.s)
        c, a = pos
        return capsule(c, a, self.r, self.l, self.s)

    def dist(self, p, pos, BOX):
        """Distance from points p to the core of the object (centre or axis segment)."""
        if self.geo == "sphere":
            return dist_to_point(p, pos, BOX)
        c, a = pos
        return dist_to_segment(p, c, a, self.l / 2 - self.r, BOX)


class Run:
    def __init__(self, run_dir, in_name, BOX, rng=1, gap=0.5):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.in_name = in_name
        self.BOX = np.asarray(BOX, float)
        self.rng = np.random.default_rng(rng)
        self.gap = gap
        self.types = []

    def random_pos(self, t):
        c = self.rng.uniform(0, 1, 3) * self.BOX
        if t.geo == "sphere":
            return c
        th = self.rng.uniform(0, np.pi)
        return c, np.array([np.cos(th), np.sin(th), 0.0])

    def overlaps(self, t, pos):
        """True if candidate t at pos touches anything already placed (any type)."""
        beads = t.beads(pos)
        for u in self.types + [t]:
            for q in u.positions:
                if u.dist(beads, q, self.BOX).min() < u.r + self.gap:
                    return True
        return False

    def add_type(self, t, max_tries=100000):
        t.atom_type = len(self.types) + 2            # 1 is cytoplasm
        tries = 0
        while len(t.positions) < t.n:
            pos = self.random_pos(t)
            if not self.overlaps(t, pos):
                t.positions.append(pos)
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"placed only {len(t.positions)} of {t.n} {t.name}; box too crowded")
        self.types.append(t)

    def fill_cytoplasm(self, RHO):
        cyto = self.rng.uniform(0, 1, (int(RHO * np.prod(self.BOX)), 3)) * self.BOX
        for t in self.types:
            for pos in t.positions:
                cyto = cyto[t.dist(cyto, pos, self.BOX) > t.r]
        self.cyto = cyto

    def write_data_file(self):
        atoms = [(0, 1, x, (0, 0, 0)) for x in self.cyto]
        mol = 0
        for t in self.types:
            for pos in t.positions:
                mol += 1                              # unique molecule ID across all types
                for x in t.beads(pos):
                    img = np.floor(x / self.BOX).astype(int)
                    atoms.append((mol, t.atom_type, x - img * self.BOX, img))
        with open(self.run_dir / f"{self.in_name}.data", "w") as f:
            f.write(f"{self.in_name}\n\n{len(atoms)} atoms\n{len(self.types) + 1} atom types\n\n")
            f.write(f"0 {self.BOX[0]} xlo xhi\n0 {self.BOX[1]} ylo yhi\n0 {self.BOX[2]} zlo zhi\n\n")
            f.write("Masses\n\n1 1.0\n")
            for t in self.types:
                f.write(f"{t.atom_type} 1.0  # {t.name}\n")
            f.write("\nAtoms # bond\n\n")
            for i, (m, ty, x, img) in enumerate(atoms, start=1):
                f.write(f"{i} {m} {ty} {x[0]:.5f} {x[1]:.5f} {x[2]:.5f} {img[0]} {img[1]} {img[2]}\n")
        return len(atoms)