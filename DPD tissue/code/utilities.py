import numpy as np
from geometry import capsule, dist_to_segment, shell, shell_bonds, sphere, dist_to_point
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
    def __init__(self, name, n, geo, r, l=None, s=0.5, rel_density=1.0, a_cyto=40.0, a_self=30.0):
        self.name, self.n, self.geo, self.r, self.l, self.s = name, n, geo, r, l, s
        self.positions = []          # sphere: c ; capsule: (c, a)
        self.atom_type = None        # set by Run.add_type
        self.mass = 3.0 * rel_density * s**3
        self.a_cyto = a_cyto                     # repulsion against cytoplasm
        self.a_self = a_self                     # repulsion between two of these

    def beads(self, pos):
        if self.geo == "sphere":
            return sphere(pos, self.r, self.s)
        if self.geo == "capsule":
            c, a = pos
            return capsule(c, a, self.r, self.l, self.s)
        if self.geo == "shell":
            return shell(pos, self.r, self.s)

    def dist(self, p, pos, BOX):
        """Distance from points p to the core of the object (centre, axis segment, or shell surface)."""
        if self.geo == "sphere":
            return dist_to_point(p, pos, BOX)
        if self.geo == "shell":
            return np.abs(dist_to_point(p, pos, BOX) - self.r)
        c, a = pos
        return dist_to_segment(p, c, a, self.l / 2 - self.r, BOX)

    def half_extent(self):
        return self.r if self.geo == "sphere" else self.l / 2




class Run:
    def __init__(self, run_dir, in_name, BOX, rng=1, gap=0.5):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.in_name = in_name
        self.BOX = np.asarray(BOX, float)
        self.rng = np.random.default_rng(rng)
        self.gap = gap
        self.types = []
        self.medium_type = None      # atom type of the fluid outside the cell (set by fill_cytoplasm)

    def random_pos(self, t):
        if t.geo == "shell":
            return self.BOX / 2
        c = self.rng.uniform(0, 1, 3) * self.BOX
        if t.geo == "sphere":
            return c
        th = self.rng.uniform(0, np.pi)
        return c, np.array([np.cos(th), np.sin(th), 0.0])

    def overlaps(self, t, pos):
        """True if candidate t at pos touches anything already placed, or sticks out of the cell."""
        beads = t.beads(pos)
        for u in self.types + [t]:
            for q in u.positions:
                if u.geo == "shell":
                    if dist_to_point(beads, q, self.BOX).max() > u.r - self.gap:
                        return True   # some bead is outside the cell or touching the membrane
                    continue
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
        fluid = self.rng.uniform(0, 1, (int(RHO * np.prod(self.BOX)), 3)) * self.BOX
        for t in self.types:
            for pos in t.positions:
                cut = t.s if t.geo == "shell" else t.r     # shell: clear a band of width s around the surface
                fluid = fluid[t.dist(fluid, pos, self.BOX) > cut]
        self.cyto, self.medium = fluid, fluid[:0]
        for t in self.types:
            if t.geo == "shell":
                inside = dist_to_point(fluid, t.positions[0], self.BOX) < t.r
                self.cyto, self.medium = fluid[inside], fluid[~inside]
                self.medium_type = len(self.types) + 2     # last atom type


    def write_data_file(self):
        atoms = [(0, 1, x, (0, 0, 0)) for x in self.cyto]
        if self.medium_type:
            atoms += [(0, self.medium_type, x, (0, 0, 0)) for x in self.medium]
        bonds, mol = [], 0
        for t in self.types:
            for pos in t.positions:
                mol += 1                              # unique molecule ID across all types
                p = t.beads(pos)
                if t.geo == "shell":
                    first = len(atoms) + 1            # global atom ID of this shell's first bead
                    i, j, d = shell_bonds(p, t.s)
                    bonds += [(1 if dd < 1.2 * t.s else 2, first + a, first + b) for a, b, dd in zip(i, j, d)]
                for x in p:
                    img = np.floor(x / self.BOX).astype(int)
                    atoms.append((mol, t.atom_type, x - img * self.BOX, img))
        n_types = len(self.types) + 1 + (self.medium_type is not None)
        with open(self.run_dir / f"{self.in_name}.data", "w") as f:
            f.write(f"{self.in_name}\n\n{len(atoms)} atoms\n{n_types} atom types\n")
            if bonds:
                f.write(f"{len(bonds)} bonds\n2 bond types\n")
            f.write(f"\n0 {self.BOX[0]} xlo xhi\n0 {self.BOX[1]} ylo yhi\n0 {self.BOX[2]} zlo zhi\n\n")
            f.write("Masses\n\n1 1.0\n")
            for t in self.types:
                f.write(f"{t.atom_type} {t.mass:.4f}  # {t.name}\n")
            if self.medium_type:
                f.write(f"{self.medium_type} 1.0  # medium\n")
            f.write("\nAtoms # bond\n\n")
            for i, (m, ty, x, img) in enumerate(atoms, start=1):
                f.write(f"{i} {m} {ty} {x[0]:.5f} {x[1]:.5f} {x[2]:.5f} {img[0]} {img[1]} {img[2]}\n")
            if bonds:
                f.write("\nBonds\n\n")
                for k, (ty, a, b) in enumerate(bonds, start=1):
                    f.write(f"{k} {ty} {a} {b}\n")
        return len(atoms)

    
    def write_input_file(self, T=1.0, a_cc=25.0, gamma=4.5, dt=0.01, steps=50000,
                     dump_every=1000, msd_every=100, thermo_every=1000, seed=34387, k_bond=100.0):
        rigid = [t for t in self.types if t.geo != "shell"]
        mem = [t for t in self.types if t.geo == "shell"]
        org = " ".join(str(t.atom_type) for t in rigid)
        mobile = " ".join(["1"] + [str(t.atom_type) for t in mem] + ([str(self.medium_type)] if self.medium_type else []))
        cutoff = max(t.half_extent() for t in rigid) + 1.5   # only rigid bodies need the long ghost cutoff
        L = ["units         lj", "atom_style    bond", "boundary      p p p",
            f"comm_modify   vel yes cutoff {cutoff:.1f}",
            "neighbor      0.3 bin", "neigh_modify  every 1 delay 0 check yes",
            f"read_data     {self.in_name}.data", "",
            f"pair_style    dpd {T} 1.0 {seed}",
            f"pair_coeff    1 1 {a_cc} {gamma}"]
        for i, t in enumerate(self.types):
            L.append(f"pair_coeff    1 {t.atom_type} {t.a_cyto} {gamma}   # cytoplasm-{t.name}")
            L.append(f"pair_coeff    {t.atom_type} {t.atom_type} {t.a_self} {gamma}")
            for u in self.types[i + 1:]:   # two different organelle types: mean of their a_self values
                L.append(f"pair_coeff    {t.atom_type} {u.atom_type} {(t.a_self + u.a_self) / 2} {gamma}")
        if self.medium_type:
            m = self.medium_type
            L.append(f"pair_coeff    1 {m} {a_cc} {gamma}   # cytoplasm-medium")
            L.append(f"pair_coeff    {m} {m} {a_cc} {gamma}")
            for t in self.types:
                L.append(f"pair_coeff    {t.atom_type} {m} {t.a_cyto} {gamma}   # {t.name}-medium")
        if mem:
            s = mem[0].s
            L += ["", "bond_style    harmonic",
                  f"bond_coeff    1 {k_bond} {s}",
                  f"bond_coeff    2 {k_bond} {s * np.sqrt(2):.4f}"]
        L += ["", "group         cyto type 1", f"group         org type {org}", f"group         mobile type {mobile}"]
        L += [f"group         {t.name} type {t.atom_type}" for t in self.types]
        L += ["neigh_modify  exclude molecule/intra org", "",
            f"velocity      mobile create {T} 4928459 mom yes rot no",
            "fix           fluid mobile nve",
            "fix           bodies org rigid/small molecule", "",
            "compute       chunks org chunk/atom molecule",
            "compute       msd org msd/chunk chunks",
            f"fix           msdout org ave/time {msd_every} 1 {msd_every} c_msd[4] file msd_org.dat mode vector", "",
            f"timestep      {dt}", f"thermo        {thermo_every}",
            f"dump          traj all custom {dump_every} dump.{self.in_name} id mol type x y z ix iy iz",
            f"run           {steps}", ""]
        (self.run_dir / f"in.{self.in_name}").write_text("\n".join(L))


def read_type_names(data_path):
    """Masses section of the data file -> {atom_type: name}, using the '# name' comments."""
    names, in_masses = {}, False
    for line in open(data_path):
        if line.strip() == "Masses":
            in_masses = True
            continue
        if in_masses and line.strip().startswith(("Atoms", "Pair", "Bond")):
            break
        if in_masses and "#" in line:
            k, rest = line.split("#", 1)
            names[int(k.split()[0])] = rest.strip()
    return names
 
 
def read_dump_com_typed(path):
    """Like read_dump_com, but also returns mol_ids[n_org] and their atom types[n_org]."""
    steps, coms = [], []
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
            else:
                pos = data[:, [c["x"], c["y"], c["z"]]] + data[:, [c["ix"], c["iy"], c["iz"]]] * box
            mol = data[:, c["mol"]].astype(int)
            ids = np.unique(mol[mol > 0])
            coms.append([pos[mol == m].mean(axis=0) for m in ids])
            if len(steps) == 1:
                types = np.array([int(data[mol == m, c["type"]][0]) for m in ids])
    return np.array(steps), np.array(coms), ids, types