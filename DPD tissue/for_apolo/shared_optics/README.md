# shared

One copy of the FB-DHS optics, used by every run folder. Keep it next to the run folders, so the layout on Apolo is

```
~/DPD_runs/
    shared/                  optics_run.py + optics_code/
    sphere_crowding_run/     make_runs.py, runs/, results/, *.sh ...
    langevin_control_run/
    <next run>/
```

Each run folder's `optics.sh` and `gather.sh` run `python ../shared/optics_run.py ...` from inside the run folder,
so `runs/manifest.csv` and `results/` are read and written there. A change to the optics (here or in
`optics_code/`) applies to every run from then on.

| File | What it is |
|---|---|
| `optics_run.py` | Tiles the sample with a level's runs, simulates the 1400 holograms (`frames`), then PSD, knee and D (`gather`). Reads `box.data` or `cell.data` for the box size |
| `optics_code/` | Copies of `DHS instrument/code/*.py` and `DPD tissue/code/tile_sample.py` |

A new run folder needs `runs/manifest.csv` (columns `run_dir`, `level`), a `traj.npz` in every run (keys `com`,
`type_names`; centres in r_c relative to the tile centre, 1860 frames), and organelle names that are in
`ORGANELLES` at the top of `optics_run.py`. Copy `optics.sh` and `gather.sh` from `langevin_control_run`.

Run folders made before this one (`crowding_run`) still use their own copy; their results were made with it.
