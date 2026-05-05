# fast-lammps

Dynamic FCC coarse-graining experiments for LAMMPS crack simulations.

The current implementation is focused on Ni FCC crack propagation:

- `spglib`-assisted lattice registration with a manual FCC fallback;
- lattice-index block tiling instead of coordinate-box heuristics;
- reversible fine/coarse mapping with explicit atom positions;
- block-level scoring and dry-run/apply execution through LAMMPS;
- memory and algorithm metrics in `logs/`.

## Environment

The project uses a local pyenv environment:

```bash
pyenv local fast_env
pip install -r requirements.txt
```

The `lammps` wheel needs MPI at runtime. `requirements.txt` includes `mpich`, and the launcher preloads its `libmpi.so.12` from the active pyenv environment.

## Test

```bash
make test
```

## Run Crack Experiment

Dry-run mode plans coarse-graining actions but does not change atoms:

```bash
python main.py --file tests/crack_ni_lg_velocity_set.in -i 10000 -m 250 -s 2
```

Apply planned actions:

```bash
python main.py --file tests/crack_ni_lg_velocity_set.in -i 10000 -m 250 -s 2 --apply-cg
```

Useful parameters:

- `--scale`: coarse lattice scale factor, default `2`.
- `--block-cells`: cubic decision block size in fine FCC conventional cells, default `2`.
- `--iteration`: total number of LAMMPS steps controlled by the launcher.
- `--measure_frequency`: LAMMPS steps between CG checks.