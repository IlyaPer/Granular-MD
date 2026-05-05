import argparse
import ctypes
import glob
import logging
import os
from pathlib import Path
import sys

from build.monitor import MemoryProfiler, create_beautiful_plot
from src.modifiers.changer import DynamicChanger
from src.utils.approximation import compute_params_CG
from src.utils.utils import LammpsCommunicator

try:
    from lammps import lammps
except ModuleNotFoundError as exc:
    raise SystemExit(
        "LAMMPS Python package is not available. Run this launcher inside the LAMMPS environment."
    ) from exc


def preload_mpi_runtime():
    mpi_lib = Path(sys.prefix) / "lib" / "libmpi.so.12"
    if mpi_lib.exists():
        ctypes.CDLL(str(mpi_lib), mode=ctypes.RTLD_GLOBAL)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Crack_Ni_CG",
        description="Dynamic FCC coarse graining for a Ni crack experiment.",
    )
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-s", "--scale", type=int, default=2)
    parser.add_argument("-i", "--iteration", type=int, required=True)
    parser.add_argument("-m", "--measure_frequency", type=int, required=True)
    parser.add_argument("--block-cells", type=int, default=2)
    parser.add_argument(
        "--dump-frequency",
        type=int,
        default=None,
        help="Override dump 1 frequency. Defaults to measure_frequency.",
    )
    parser.add_argument(
        "--apply-cg",
        action="store_true",
        help="Apply planned CG actions. Default is dry-run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lattice_constant = 3.52
    sigma_cg, lattice_cg, epsilon_cg, mass_cg = compute_params_CG(args.scale)

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/dynamic_coarse_graining.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(
        "CG params: sigma=%s lattice=%s epsilon=%s mass=%s",
        sigma_cg,
        lattice_cg,
        epsilon_cg,
        mass_cg,
    )

    preload_mpi_runtime()
    solver = lammps()
    solver.file(args.file)
    configure_dump(solver, args.dump_frequency or args.measure_frequency)

    communicator = LammpsCommunicator(solver)
    changer = DynamicChanger(
        communicator=communicator,
        lattice_constant=lattice_constant,
        scale_factor=args.scale,
        block_cells=args.block_cells,
        dry_run=not args.apply_cg,
    )

    solver.commands_string(
        f"""
pair_coeff      1 2 20 2.28
pair_coeff      2 2 {epsilon_cg} {sigma_cg}
mass            2 {mass_cg}
lattice         fcc {lattice_cg}
"""
    )

    step = 0
    with MemoryProfiler(name="accelerate", track_objects=True, snapshot_interval=20) as profiler:
        while step < args.iteration:
            profiler.snapshot(iteration=step, label=f"before_step_{step}")
            solver.cmd.run(args.measure_frequency)
            changer.accelerate(solver)
            solver.command("reset_atoms id")
            solver.command("run 0")
            step += args.measure_frequency
            profiler.snapshot(iteration=step, label=f"after_step_{step}")
            logging.info("Current iteration: %s", step)

    final_dump = "logs/final_state.lammpstrj"
    solver.command(f"write_dump all custom {final_dump} id type x y z modify sort id")

    metric_files = glob.glob("logs/metrics_*.json")
    if metric_files:
        latest_file = max(metric_files, key=os.path.getctime)
        create_beautiful_plot(latest_file, "logs/memory_analysis.png")
    print("Trajectory dump: dump_accurate.crack_GRAIN.lammpstrj")
    print(f"Final snapshot: {final_dump}")
    print("Memory plot: logs/memory_analysis.png")


def configure_dump(solver, frequency):
    if frequency < 1:
        raise ValueError("dump frequency must be positive.")
    try:
        solver.command(f"dump_modify 1 every {frequency} first yes sort id")
    except Exception:
        logging.info("Input file has no dump with id 1; only final snapshot will be written.")


if __name__ == "__main__":
    main()