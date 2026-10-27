"""
Compress the LAMMPS output of each run into the traj.npz that tile_sample and the optics read.

Usage:
    python compress_dumps.py runs/phi10/cell_0 runs/phi10/cell_1 ...
    python compress_dumps.py --manifest runs/manifest.csv --task $SLURM_ARRAY_TASK_ID
    python compress_dumps.py --manifest runs/manifest.csv            (every run)

Two kinds of run folder are understood:
  crowding runs (make_runs.py): cell.data, com_org.dat (organelle centres every frame),
      com_mem.dat (membrane centre every frame) and dump.membrane (membrane beads every MEM_EVERY frames)
  older runs: cell.data and dump.cell (membrane + organelle beads every frame)
The output is <run>/traj.npz with
    steps      (n_frames,)                 timestep of each frame
    com        (n_frames, n_org, 3)        organelle centres of mass, relative to the membrane centre (r_c)
    org_type   (n_org,)                    atom type of each organelle
    type_names (n_org,)                    name of each organelle, from the Masses comments of cell.data
    mem        (n_mem_frames, n_mem, 3)    membrane beads, relative to the membrane centre (r_c)
    mem_steps  (n_mem_frames,)             timestep of each membrane frame (crowding runs only)
the same keys as compress_dump in DPD tissue/code/utilities.py, so old and new runs load the same way.
"""
import argparse
import csv
from itertools import islice
from pathlib import Path

import numpy as np


def read_type_names(data_path):
    """
    Read the organelle names from the Masses section of a LAMMPS data file.

    Parameters:
    data_path : str or Path
        LAMMPS data file whose Masses lines end with '# name'.

    Returns:
    names : dict
        Name of each atom type, {atom_type: name}.
    """
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


def read_frames(path):
    """
    Iterate over the frames of a LAMMPS custom dump.

    Parameters:
    path : str or Path
        Dump written with 'dump ... custom N file id mol type x y z ix iy iz'.

    Returns:
    (step, box, cols, data) per frame : generator
        Timestep, box lengths (3,), column names and the (n_atoms, n_cols) array.
        A frame cut short at the end of the file (job killed while writing) is skipped.
    """
    with open(path) as f:
        while f.readline().startswith("ITEM: TIMESTEP"):
            step = int(f.readline())
            f.readline(); n = int(f.readline())
            f.readline()
            box = np.array([np.ptp([float(v) for v in f.readline().split()[:2]]) for _ in range(3)])
            cols = f.readline().split()[2:]
            lines = list(islice(f, n))
            if len(lines) < n:
                return

            # One split over the whole frame is ~10x faster than np.loadtxt
            data = np.array(" ".join(lines).split(), dtype=float).reshape(n, len(cols))
            yield step, box, cols, data


def read_atom_types(data_path):
    """
    Molecule id and atom type of every atom, from the Atoms section of a LAMMPS data file.

    Parameters:
    data_path : str or Path
        LAMMPS data file written by Run.write_data_file (Atoms # bond: id mol type x y z ix iy iz).

    Returns:
    mol, typ : 1D int arrays
        Molecule id and atom type, in the order of the file.
    """
    with open(data_path) as f:
        for line in f:
            if line.startswith("Atoms"):
                break
        f.readline()
        rows = []
        for line in f:
            if not line.strip():
                break
            rows.append(line.split(None, 3)[1:3])
    a = np.array(rows, dtype=int)
    return a[:, 0], a[:, 1]


def read_ave_time(path):
    """
    Read the file of a 'fix ave/time' (scalar or vector mode).

    Parameters:
    path : str or Path
        File written by fix ave/time ... file <path> [mode vector].

    Returns:
    steps : 1D int array
        Timestep of each output.
    values : 2D or 3D array
        (n_out, n_values) in scalar mode, (n_out, n_rows, n_values) in vector mode (the row index column is dropped).
    """
    lines = [l for l in open(path) if not l.startswith("#")]
    first = lines[0].split()
    if len(first) != 2:                                   # scalar mode: one line per output
        a = np.array(" ".join(lines).split(), dtype=float).reshape(len(lines), -1)
        return a[:, 0].astype(int), a[:, 1:]

    # Vector mode: a 'step n_rows' line followed by n_rows lines 'row v1 v2 ...'
    n = int(first[1])
    block = n + 1
    n_out = len(lines) // block                           # a block cut short at the end is dropped
    steps = np.array([int(lines[i * block].split()[0]) for i in range(n_out)])
    body = [l for i in range(n_out) for l in lines[i * block + 1:(i + 1) * block]]
    vals = np.array(" ".join(body).split(), dtype=float).reshape(n_out, n, -1)
    return steps, vals[:, :, 1:]


def compress_com(run_dir, name="cell", membrane="membrane", keep_mem=True):
    """
    Turn com_org.dat, com_mem.dat and dump.membrane of a crowding run into traj.npz.

    Parameters:
    run_dir : str or Path
        Run folder from make_runs.py.
    name : str, optional
        Name of the data file. Default is 'cell'.
    membrane : str, optional
        Name of the membrane type in the Masses comments. Default is 'membrane'.
    keep_mem : bool, optional
        Also store the membrane beads of dump.membrane. Default is True.

    Returns:
    out : Path
        Path of the traj.npz written.
    """
    run_dir = Path(run_dir)
    names = read_type_names(run_dir / f"{name}.data")
    mem_type = [t for t, n in names.items() if n == membrane][0]

    # Organelle of each chunk: 'chunk/atom molecule compress yes' numbers the molecules of the
    # org group (every non-membrane type that is in a molecule) in increasing molecule id
    mol, typ = read_atom_types(run_dir / f"{name}.data")
    org = (mol > 0) & np.isin(typ, [t for t, n in names.items() if n not in (membrane, "medium")])
    ids, first = np.unique(mol[org], return_index=True)
    org_type = typ[org][first]

    steps, com = read_ave_time(run_dir / "com_org.dat")
    msteps, centre = read_ave_time(run_dir / "com_mem.dat")
    if com.shape[1] != len(ids):
        raise ValueError(f"{run_dir}: com_org.dat has {com.shape[1]} rows, cell.data has {len(ids)} organelles")

    # Keep the frames present in both files, relative to the membrane centre
    common, i, j = np.intersect1d(steps, msteps, return_indices=True)
    com = com[i] - centre[j][:, None, :]

    extra = {}
    if keep_mem and (run_dir / "dump.membrane").exists():
        cmap = dict(zip(msteps, centre))
        ms, mems = [], []
        for step, box, cols, data in read_frames(run_dir / "dump.membrane"):
            if step not in cmap:
                continue
            c = {k: n for n, k in enumerate(cols)}
            data = data[np.argsort(data[:, c["id"]])]
            pos = data[:, [c["x"], c["y"], c["z"]]] + data[:, [c["ix"], c["iy"], c["iz"]]] * box
            ms.append(step); mems.append(pos - cmap[step])
        extra = dict(mem=np.array(mems, np.float32), mem_steps=np.array(ms))

    out = run_dir / "traj.npz"
    np.savez_compressed(out, steps=common, com=com.astype(np.float32), org_type=org_type,
                        type_names=np.array([names[t] for t in org_type]), **extra)
    return out


def compress_dump(run_dir, name="cell", membrane="membrane", keep_mem=True):
    """
    Turn dump.<name> into traj.npz: one centre of mass per organelle and every membrane bead per frame.

    Parameters:
    run_dir : str or Path
        Run folder with <name>.data and dump.<name>.
    name : str, optional
        Name of the run files. Default is 'cell'.
    membrane : str, optional
        Name of the membrane type in the Masses comments. Default is 'membrane'.
    keep_mem : bool, optional
        Store the membrane beads (about 90% of the file). The optics only needs com. Default is True.

    Returns:
    out : Path
        Path of the traj.npz written.
    """
    run_dir = Path(run_dir)
    names = read_type_names(run_dir / f"{name}.data")
    mem_type = [t for t, n in names.items() if n == membrane][0]

    steps, coms, mems = [], [], []
    for step, box, cols, data in read_frames(run_dir / f"dump.{name}"):
        c = {k: i for i, k in enumerate(cols)}
        data = data[np.argsort(data[:, c["id"]])]                       # same bead order every frame

        # Unwrapped positions, relative to the membrane centre so the cell stays centred in its tile
        pos = data[:, [c["x"], c["y"], c["z"]]] + data[:, [c["ix"], c["iy"], c["iz"]]] * box
        mol, typ = data[:, c["mol"]].astype(int), data[:, c["type"]].astype(int)
        is_mem = typ == mem_type
        centre = pos[is_mem].mean(axis=0)

        # Centre of mass of each organelle (rigid body = one molecule id)
        if not steps:
            ids = np.unique(mol[~is_mem])
            order = np.argsort(mol[~is_mem], kind="stable")
            counts = np.bincount(np.searchsorted(ids, mol[~is_mem]))
            org_type = np.array([typ[mol == m][0] for m in ids])
        p = pos[~is_mem][order]
        com = np.add.reduceat(p, np.r_[0, np.cumsum(counts)[:-1]], axis=0) / counts[:, None]

        steps.append(step)
        coms.append(com - centre)
        if keep_mem:
            mems.append(pos[is_mem] - centre)

    out = run_dir / "traj.npz"
    extra = dict(mem=np.array(mems, np.float32)) if keep_mem else {}
    np.savez_compressed(out, steps=np.array(steps), com=np.array(coms, np.float32),
                        org_type=org_type, type_names=np.array([names[t] for t in org_type]), **extra)
    return out


def run_path(manifest, run_dir):
    """
    Folder of a manifest row. make_runs.py writes run_dir relative to crowding_run (runs/phi10/cell_0),
    so it is looked up next to the runs folder first, then next to the manifest.

    Parameters:
    manifest : str or Path
        runs/manifest.csv.
    run_dir : str
        The run_dir column of the row.

    Returns:
    path : Path
        Absolute path of the run folder.
    """
    m = Path(manifest).resolve().parent
    for base in (m.parent, m):
        if (base / run_dir).is_dir():
            return base / run_dir
    raise FileNotFoundError(f"{run_dir} is neither in {m.parent} nor in {m}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders")
    ap.add_argument("--manifest", help="CSV with a run_dir column (paths relative to the CSV's folder)")
    ap.add_argument("--task", type=int, help="row of the manifest to compress (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--name", default="cell")
    ap.add_argument("--no-mem", action="store_true", help="drop the membrane beads")
    a = ap.parse_args()

    runs = [Path(r) for r in a.runs]
    if a.manifest:
        rows = list(csv.DictReader(open(a.manifest)))
        sel = rows if a.task is None else [rows[a.task]]
        runs += [run_path(a.manifest, r["run_dir"]) for r in sel]

    for r in runs:
        if (r / "com_org.dat").exists():
            out = compress_com(r, a.name, keep_mem=not a.no_mem)
        else:
            out = compress_dump(r, a.name, keep_mem=not a.no_mem)
        d = np.load(out)
        print(f"{out}: {len(d['steps'])} frames, {d['com'].shape[1]} organelles, {out.stat().st_size / 1e6:.0f} MB")
