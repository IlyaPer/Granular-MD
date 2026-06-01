from lammps import lammps
import numpy as np
import argparse
import logging
from src.extractors.extractors import INFERENCE, RANDOM_CONDITION, FccCellsExtractor
from src.modifiers.changer import DynamicChanger
from src.utils.utils import LammpsCommunicator

parser = argparse.ArgumentParser(
    prog="fast lammps package",
    description="acceleration molecular dynamics with dynamic coarse-graining",
    epilog="Text at the bottom of help",
)

parser.add_argument("-f", "--file")
parser.add_argument("-a", "--mass_scale_factor")
parser.add_argument("-b", "--symmetry_extending_factor")
parser.add_argument("-c", "--potential_scale_factor")
parser.add_argument("-m", "--measure_frequency")
parser.add_argument("-l", "--log_interval")
parser.add_argument("--accelerate")
parser.add_argument("--smoke_test")
parser.add_argument("-i", "--max_iter")
parser.add_argument("--symmetry")
parser.add_argument("--lattice_constant")
parser.add_argument("--max_granulate_factor")

args = parser.parse_args()

logging.basicConfig(
    filename="records/fast_lammps_run_DATE.log", # TODO: set date
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


lmp = lammps()

lmp.file(args.file)

args.smoke_test = RANDOM_CONDITION

communicator = LammpsCommunicator(lmp)
solver = FccCellsExtractor(communicator, float(args.lattice_constant), smoke_test=args.smoke_test, scale_factor=int(args.symmetry_extending_factor))
dc = DynamicChanger(communicator, solver, float(args.lattice_constant), scale_factor=int(args.symmetry_extending_factor), baby_mode=False)


iteration = 0
while iteration < int(args.max_iter):
    dc.accelerate(lmp)
    lmp = dc.communicator.get_instance()
    lmp.command(f"run {int(args.measure_frequency)}")

    communicator = LammpsCommunicator(lmp)
    solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
    dc = DynamicChanger(
        communicator, solver, 3.52, scale_factor=2, baby_mode=True
    )
    iteration += int(args.measure_frequency)
