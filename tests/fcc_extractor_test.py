import unittest
import ase
from src.extractors.extractors import *
from src.utils.approximation import compute_params_CG
from src.utils.utils import LammpsCommunicator
from src.modifiers.changer import DynamicChanger
from lammps import lammps
import numpy as np


class TestFccCellsExtractionSimple(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/TEST_crack_ni_lg_velocity_set.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A)
        self.dc = DynamicChanger(self.communicator, self.solver, A, A_CG, baby_mode=True)

        block = f"""
        pair_coeff      1 2 20 2.28
        pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        mass            2 {ATOMIC_UNIT_MASS_CG}
        lattice         fcc {A_CG}
        """

        self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

        self.L.cmd.run(500)

    def test_simple_split_1_mega_cell(self):
        self.dc.accelerate(self.L)
        _, ids = self.solver._get_cell_ids(self.xlo, self.ylo, self.zlo)

        self.assertEqual(len(ids), 32, "incorrect number of atoms")

    def test_simple_split_2_mega_cell(self):
        self.dc.accelerate(self.L)
        cell_borders, ids = self.solver._get_cell_ids(self.xlo, self.ylo, self.zlo)
        type_of_cell = self.solver._process_single_cell(ids, cell_borders)

        self.assertEqual(type_of_cell, SIMPLE, "incorrect type of mega cell")


class TestFccCellsExtractionWithCrack(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/TEST_crack_ni_lg_velocity_set_2.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A)
        self.dc = DynamicChanger(self.communicator, self.solver, A, A_CG, baby_mode=True)

        block = f"""
        pair_coeff      1 2 20 2.28
        pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        mass            2 {ATOMIC_UNIT_MASS_CG}
        lattice         fcc {A_CG}
        """

        self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

    def test_basic_functionality_not_empty(self):
        ids = self.communicator.__get_atom_identificators__()
        types = self.communicator.__get_atom_types__()

        self.assertEqual(len(ids) > 30, True, f"Only {len(ids)}.")

        self.assertEqual(len(types) > 30, True, f"Only {len(types)}.")

    def test_simple_split_1_mega_cell(self):
        self.L.command("run 1")

        _, ids = self.solver._get_cell_ids(self.xlo, self.ylo, self.zlo)

        self.assertEqual(
            len(ids) < 15,
            True,
            f"incorrect number of atoms: should be lower than {len(ids)}",
        )

    def test_simple_split_1_mega_cell_after_run(self):
        self.L.cmd.run(500)

        _, ids = self.solver._get_cell_ids(self.xlo, self.ylo, self.zlo)

        self.assertEqual(
            len(ids) < 15,
            True,
            f"incorrect number of atoms: should be lower than {len(ids)}",
        )

    def test_simple_split_identification(self):
        cell_borders, ids = self.solver._get_cell_ids(self.xlo, self.ylo, self.zlo)
        type_of_cell = self.solver._process_single_cell(ids, cell_borders)

        self.assertEqual(
            type_of_cell,
            CRACK,
            f"incorrect type of mega cell: should be crack, not {type_of_cell}",
        )

    def test_simple_split_identification_after_run(self):
        self.L.cmd.run(500)
        cell_borders, ids = self.solver._get_cell_ids(self.xlo, self.ylo, self.zlo)
        type_of_cell = self.solver._process_single_cell(ids, cell_borders)

        self.assertEqual(
            type_of_cell,
            CRACK,
            f"incorrect type of mega cell: should be crack, not {type_of_cell}",
        )


class TestAllRegionsApproximate(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/TEST_BIG_crack_ni_lg_velocity_set.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A)
        self.dc = DynamicChanger(self.communicator, self.solver, A, A_CG, baby_mode=True)

        block = f"""
        pair_coeff      1 2 20 2.28
        pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        mass            2 {ATOMIC_UNIT_MASS_CG}
        lattice         fcc {A_CG}
        """

        self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

    def test_identification_all_regions(self):
        # TEST WHERE ALL REGIONS ARE TO BE APPROXIMATED

        self.solver.extract_interesting_regions()

        self.assertEqual(
            len(self.solver._get_cells_to_approximate()) > 10,
            True,
            f"incorrect number of cells to approximate. Should be > 10, not {len(self.solver._get_cells_to_approximate())}",
        )

        for _, ids in self.solver._get_cells_to_approximate():
            self.assertEqual(
                len(ids) > 20,
                True,
                f"incorrect number of atoms in cells to approximate. Should be at least 20, not {len(ids)}",
            )


class TestApproximation(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/TEST_granulate.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(
            self.communicator, A, lower_threshold=-2.5, upper_threshold=-1
        )
        self.dc = DynamicChanger(self.communicator, self.solver, A, A_CG, baby_mode=True)

        block = f"""
        pair_coeff      1 2 20 2.28
        pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        mass            2 {ATOMIC_UNIT_MASS_CG}
        lattice         fcc {A_CG}
        """

        self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

    def test_identification_all_regions(self):
        # TEST WHERE ALL REGIONS ARE TO BE APPROXIMATED

        self.L.command("run 1000")

        self.solver.extract_interesting_regions()

        self.assertEqual(
            len(self.solver._get_cells_to_approximate()),
            1,
            f"incorrect number of cells to approximate. Should be 1, not {len(self.solver._get_cells_to_approximate())}",
        )

        # for _, ids in self.solver._get_cells_to_approximate():
        #     self.assertEqual(len(ids) > 20, True,
        #                  f'incorrect number of atoms in cells to approximate. Should be at least 20, not {len(ids)}')

    def test_approximate(self):
        # TEST WHERE ALL REGIONS ARE TO BE APPROXIMATED

        self.L.command("run 1000")

        self.dc.accelerate(self.L)

        self.solver.extract_interesting_regions()

        L2 = self.dc._lammps_execute()
        self.assertEqual(
            self.L,
            L2,
            f"different instances ",
        )

        self.assertEqual(
            len(self.solver._get_cells_to_approximate()) == 1,
            True,
            f"There should be no regions to be approximated, found {len(self.solver._get_cells_to_approximate())}",
        )

        # self.assertEqual(len(self.solver.debug_cells_grained) > 10, True,
        #                  f'There should be approximated regions, found {len(self.solver._get_cells_to_approximate())}')

        # for _, ids in self.solver.debug_cells_grained:
        #     self.assertEqual(len(ids) <= 4, True,
        #                  f'incorrect number of atoms in grained cells. Should be no more than 4, found {4}')


class TestGranulation(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/TEST_granulate.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi, self.pbc = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, lower_threshold=-2.5)
        self.dc = DynamicChanger(self.communicator, self.solver, A, A_CG, baby_mode=False)

        # block = f"""
        # pair_coeff      1 2 20 2.28
        # pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        # mass            2 {ATOMIC_UNIT_MASS_CG}
        # lattice         fcc {A_CG}
        # """

        # self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

    def test_region_to_granulate(self):
        cell_size = 7.04

        nx = int(np.floor((self.xhi - self.xlo) / cell_size))
        ny = int(np.floor((self.yhi - self.ylo) / cell_size))
        nz = int(np.floor((self.zhi - self.zlo) / cell_size))

        nx -= 3
        ny -= 3
        nz -= 3

        cell, ids_of_the_cell = self.solver._get_cell_ids(nx, ny, nz)
        self.assertEqual(
            len(ids_of_the_cell) > 20,
            True,
            f"incorrect number of atoms: should be lower than {len(ids_of_the_cell)}",
        )
        type_of_cell = self.solver._process_single_cell(ids_of_the_cell, cell)
        self.assertEqual(type_of_cell, SIMPLE)
        self.assertEqual(
            self.solver._solver_rule(ids_of_the_cell, to_approximate=True), True
        )

    def test_granulation(self):
        self.dc.accelerate(self.L)

        L2 = self.dc._lammps_execute()
        self.assertEqual(
            self.L,
            L2,
            f"different instances ",
        )

        self.assertEqual(
            len(self.solver._get_cells_to_approximate()),
            0,
            f"There should be no enough regions to be approximated, found only {len(self.solver._get_cells_to_approximate())}",
        )
        self.assertEqual(
            len(self.solver._get_cells_to_granulate()) > 2,
            True,
            f"There should be no enough regions to be GRANULATED, found only {len(self.solver._get_cells_to_granulate())}",
        )

        # self.assertEqual(
        #     len(self.solver._get_cells_to_approximate()),
        #     len(self.dc._get_debug_info()),
        #     f"There should be equal number of grained and to be grained cells {len(self.solver._get_cells_to_approximate())}",
        # ) #TODO: check why it doesn't pass
        self.L.command("reset_atoms id")

        self.L.command("run 2000")
        # self.communicator = LammpsCommunicator(self.L)
        # self.dc.set_lammps(self.communicator)

        self.dc.accelerate(self.L)


        for borders_of_grained_cell in self.dc._get_debug_info():
            ids_in_grained_cells = self.dc.communicator._extract_ids_from_block(borders_of_grained_cell)
            self.assertEqual(
                len(ids_in_grained_cells),
                14,
                f"incorrect number of atoms in grained cells. Should be 4, found {len(ids_in_grained_cells)}",
            )
        # TODO: law of masses, velocities distribution!!!


class TestGranulationFluctuations(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/TEST_granulate_with_fluctuations.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi, self.pbc = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, lower_threshold=-2.5, upper_threshold=-1.7)
        self.dc = DynamicChanger(self.communicator, self.solver, A, A_CG, baby_mode=False)

        # block = f"""
        # pair_coeff      1 2 20 2.28
        # pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        # mass            2 {ATOMIC_UNIT_MASS_CG}
        # lattice         fcc {A_CG}
        # """

        # self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

    def test_granulation_with_fluctuations(self):
        

        for i in range(50):

            self.L.command("run 250")
            self.dc.accelerate(self.L)
            self.L.command("reset_atoms id")

            if len(self.solver._get_cells_to_granulate()) > 0:
                for borders_of_grained_cell in self.dc._get_debug_info():
                    ids_in_grained_cells = self.dc.communicator._extract_ids_from_block(borders_of_grained_cell)
                    self.assertEqual(
                        len(ids_in_grained_cells),
                        14,
                        f"incorrect number of atoms in grained cells. Should be 4, found {len(ids_in_grained_cells)}",
                    )


if __name__ == "__main__":
    unittest.main()
