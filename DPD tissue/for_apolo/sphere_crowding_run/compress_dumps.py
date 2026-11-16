"""
Compress the LAMMPS output of each sphere run into the traj.npz that tile_sample and the optics read.

Usage:
    python compress_dumps.py runs/phi10/box_0 runs/phi10/box_1 ...
    python compress_dumps.py --manifest runs/manifest.csv --task $SLURM_ARRAY_TASK_ID
    python compress_dumps.py --manifest runs/manifest.csv            (every run)

A run folder (make_runs.py) has box.data and com_org.dat (sphere centres every frame). There is no
membrane, so the centres are taken relative to the box centre, which puts the box in the middle of its tile.
The output is <run>/traj.npz with
    steps      (n_frames,)                 timestep of each frame
    com        (n_frames, n_org, 3)        sphere centres of mass (unwrapped), relative to the box centre (r_c)
    org_type   (n_org,)                    atom type of each sphere
    type_names (n_org,)                    name of each sphere type, from the Masses comments of box.data
the same keys as crowding_run (without mem), so the optics loads both the same way.
"""
import argparse
import csv
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


def box_length(data_path):
    """
    Box lengths of a LAMMPS data file.

    Parameters:
    data_path : str or Path
        LAMMPS data file.

    Returns:
    L : 1D array
        Box length along x, y and z.
    """
    L = {}
    for line in open(data_path):
        for ax in "xyz":
            if f"{ax}lo {ax}hi" in line:
                lo, hi = map(float, line.split()[:2])
                L[ax] = hi - lo
        if len(L) == 3:
            return np.array([L["x"], L["y"], L["z"]])
    raise ValueError(f"no box lines in {data_path}")


def compress_spheres(run_dir, name="box"):
    """
    Turn com_org.dat of a sphere run into traj.npz.

    Parameters:
    run_dir : str or Path
        Run folder from make_runs.py.
    name : str, optional
        Name of the data file. Default is 'box'.

    Returns:
    out : Path
        Path of the traj.npz written.
    """
    run_dir = Path(run_dir)
    names = read_type_names(run_dir / f"{name}.data")

    # Sphere of each chunk: 'chunk/atom molecule compress yes' numbers the molecules in increasing molecule id
    mol, typ = read_atom_types(run_dir / f"{name}.data")
    org = mol > 0
    ids, first = np.unique(mol[org], return_index=True)
    org_type = typ[org][first]

    steps, com = read_ave_time(run_dir / "com_org.dat")
    if com.shape[1] != len(ids):
        raise ValueError(f"{run_dir}: com_org.dat has {com.shape[1]} rows, {name}.data has {len(ids)} spheres")
    com = com - box_length(run_dir / f"{name}.data") / 2

    out = run_dir / "traj.npz"
    np.savez_compressed(out, steps=steps, com=com.astype(np.float32), org_type=org_type,
                        type_names=np.array([names[t] for t in org_type]))
    return out


def run_path(manifest, run_dir):
    """
    Folder of a manifest row. make_runs.py writes run_dir relative to sphere_crowding_run (runs/phi10/box_0),
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
    ap.add_argument("--name", default="box")
    a = ap.parse_args()

    runs = [Path(r) for r in a.runs]
    if a.manifest:
        rows = list(csv.DictReader(open(a.manifest)))
        sel = rows if a.task is None else [rows[a.task]]
        runs += [run_path(a.manifest, r["run_dir"]) for r in sel]

    for r in runs:
        out = compress_spheres(r, a.name)
        d = np.load(out)
        print(f"{out}: {len(d['steps'])} frames, {d['com'].shape[1]} spheres, {out.stat().st_size / 1e6:.0f} MB")
