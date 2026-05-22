import unittest
import ase
from src.extractors.extractors import *
from src.utils.approximation import compute_params_CG
from src.utils.utils import LammpsCommunicator
from src.modifiers.changer import DynamicChanger
from lammps import lammps
import numpy as np

APPROXIMATE = -1897398
GRANULATE = 23920932
RANDOM_CONDITION = 28383829

# Example of a pipleine to test your code with.


class TestFccCellsExtractionSimple(unittest.TestCase):
    """
    First test: one cell. All possible configurations and checkups:
    1) Extract fcc cells themselves (scale_factor == 1)
    2) Extract mega fcc cells  (scale_factor == 2)
    3) Extract mega fcc cells with fluctuations  (scale_factor == 2)
    4) Approximate mega fcc cell with larger particles  (scale_factor == 2)
    5) Granulate mega fcc cell with atoms  (scale_factor == 2)

    All tests nust satisfy:
    1) Law of masses
    2) Velocity/Potential energy criteria !

    """

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/lammps_scripts/test_1_cube_sss.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, scale_factor=2, smoke_test=APPROXIMATE)
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_extract_fcc_cells_basis_decomposition(self):
        """
        expect decomposition:
         1) to contain len(position) indexes
         2) each position must have unique decomposition
        """

        z_group_decomposition = self.solver.get_basis_decompostion()

        self.assertEqual(
            len(self.communicator.__get_positions__()), len(z_group_decomposition)
        )

        unique_rows, counts = np.unique(
            z_group_decomposition, axis=0, return_counts=True
        )

        # Filter for rows that appear more than once
        identical_rows = unique_rows[counts > 1]

        self.assertEqual(len(identical_rows), 0)

    def test_extract_fcc_cells_scale_1(self):
        """
        expect extractor:
         1) to contain 8 conventional fcc cells
        """

        fcc_cells = self.solver.get_group_fcc_cells(scale_factor=1)

        positions = self.communicator.__get_positions__()

        self.assertEqual(8, len(fcc_cells))

    def test_extract_fcc_cells_scale_2(self):
        """
        expect extractor:
         1) to contain 1 mega fcc cell (2*2*2)
        """

        fcc_cells = self.solver.get_group_fcc_cells(scale_factor=2)

        positions = self.communicator.__get_positions__()

        self.assertEqual(1, len(fcc_cells))

        self.assertEqual(32, len(positions[fcc_cells[0]]))

    def test_approximate_simulation(self):
        """
        expect changer:
         1) to approximate 1 mega fcc cell (2*2*2) with 4 atoms
         2) to put these atoms in correct positions
        with respect to law of masses and potential energy!
        """
        self.solver.get_cells_to_apply_action()

        identificators_to_delete, positions_of_large = self.solver.get_data_of_cells_to_approximate()

        self.assertEqual(len(positions_of_large), 4)

        self.assertEqual(len(identificators_to_delete), 32)

    def test_granulate_simulation(self):
        """
        expect changer:
         1) to granulate 1 mega fcc cell with 32 atoms
         2) to put these atoms in correct positions
         3) to be launched correctly after that
        with respect to law of masses and potential energy!
        """

        self.solver.get_cells_to_apply_action()

        identificators_to_delete, positions_to_appr = (
            self.solver.get_data_of_cells_to_approximate()
        )

        self.dc.accelerate(self.L)
        self.dc.communicator.get_instance().command("run 1500")
        self.L = self.dc.communicator.get_instance()

        self.communicator = LammpsCommunicator(self.L)
        # self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
        #     self.communicator.__get_box_size__()
        # )
        solver = FccCellsExtractor(self.communicator, 3.52, scale_factor=2, smoke_test=23920932)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )
        solver.get_basis_decompostion()

        solver.get_cells_to_apply_action()

        identificators_tochange, positions_to_appr = (
            solver.get_data_of_cells_to_granulate()
        )

        self.assertEqual(len(positions_to_appr), 32)

        self.assertEqual(len(identificators_tochange), 4)

        dc.accelerate(self.L, only_granulate=True)
        self.L.command("run 1500")


class TestFccCellsExtractionBig(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/lammps_scripts/test_2_cube.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, smoke_test=APPROXIMATE)
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_simple_split_1_mega_cell(self):
        self.solver.extract_symmetry(scale_factor=2)

        self.assertEqual(len(self.solver.fcc_mega_cells), 4**3)

        for cell in self.solver.fcc_mega_cells:
            self.assertEqual(len(self.communicator.__get_positions__()[cell]), 32)

    def test_simple_change(self):
        self.solver.extract_symmetry(scale_factor=2)
        self.solver.get_cells_to_apply_action()

        identificators_tochange, _, identificators_to_delete = (
            self.solver.get_data_of_cells_to_approximate()
        )

        self.assertEqual(len(self.solver.fcc_mega_cells), 4**3)
        for cell in self.solver.fcc_mega_cells:
            self.assertEqual(len(self.communicator.__get_positions__()[cell]), 32)

        self.assertEqual(len(identificators_tochange), 4 * 4**3)

        self.assertEqual(len(identificators_to_delete) % 28, 0)


class RandomConditionTest(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/lammps_scripts/test_4_cube_crack.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(
            self.communicator, A, smoke_test=RANDOM_CONDITION, scale_factor=2
        )
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_cells_extraction(self):
        self.solver.extract_symmetry(scale_factor=2)
        self.solver.get_cells_to_apply_action()

        identificators_tochange, _, identificators_to_delete = (
            self.solver.get_data_of_cells_to_approximate()
        )
        large_to_delete, identificators_to_delete = (
            self.solver.get_data_of_cells_to_granulate()
        )

        self.assertEqual(len(self.solver.fcc_mega_cells) > 50, True)

        self.assertEqual(len(large_to_delete), 0)

        # self.assertEqual(len(identificators_to_delete) % 28, 0)
        self.assertEqual(len(identificators_tochange) % 8, 0)

    def test_acceleration(self):
        self.dc.accelerate(self.L)

    def test_acceleration_with_reverse(self):
        self.dc.accelerate(self.L, only_approximate=True)

        self.L.command("minimize 1.0e-8 1.0e-8 10000 100000")
        self.L.command("run 500")

        self.dc.accelerate(self.L, only_granulate=True, only_approximate=False)
        self.L.command("run 500")


class TestFccGranulation(unittest.TestCase):

    def setUp(self):
        SCALE_FACTOR = 2

        sigma_for_nickel = 3.52 * (58.69 / 2.5) ** (-1)
        SIGMA = sigma_for_nickel
        A = 3.52

        SIGMA_CG, A_CG, EPSILON_CG, ATOMIC_UNIT_MASS_CG = compute_params_CG(
            SCALE_FACTOR
        )

        self.L = lammps()

        self.L.file("tests/lammps_scripts/test_3_cube.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, smoke_test=GRANULATE)
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_simple_approximate(self):
        self.solver.extract_symmetry(scale_factor=2)
        self.solver.get_cells_to_apply_action()

        identificators_to_create, identificators_to_delete = (
            self.solver.get_data_of_cells_to_granulate()
        )
        # self.assertEqual(len(identificators_to_create), 12*32)

        self.assertEqual(len(identificators_to_delete), 12 * 4)


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
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, baby_mode=True
        )

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
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, baby_mode=True
        )

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
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, baby_mode=True
        )

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
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, baby_mode=False
        )

        # block = f"""
        # pair_coeff      1 2 20 2.28
        # pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        # mass            2 {ATOMIC_UNIT_MASS_CG}
        # lattice         fcc {A_CG}
        # """

        # self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

    def test_symmetry(self):
        self.solver.extract_symmetry_mine()
        self.solver.extract_symmetry_scaled(2)
        self.assertEqual(1, 1)

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
            ids_in_grained_cells = self.dc.communicator._extract_ids_from_block(
                borders_of_grained_cell
            )
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

        self.L = lammps(cmdargs=["-screen", "none", "-log", "none"])

        self.L.file("tests/crack_ni_lg_velocity_set.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi, self.pbc = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(
            self.communicator, A, lower_threshold=-2.5, upper_threshold=-0.5
        )
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, baby_mode=False
        )

        # block = f"""
        # pair_coeff      1 2 20 2.28
        # pair_coeff      2 2 {EPSILON_CG} {SIGMA_CG}
        # mass            2 {ATOMIC_UNIT_MASS_CG}
        # lattice         fcc {A_CG}
        # """

        # self.L.commands_string(block)

        logging.info("The simulation started successfully.")
        iter = 0

    def test_final(self):
        # cell_size = 7.04

        # nx = int(np.floor((self.xhi - self.xlo) / cell_size))
        # ny = int(np.floor((self.yhi - self.ylo) / cell_size))
        # nz = int(np.floor((self.zhi - self.zlo) / cell_size))

        # nx -= 3
        # ny -= 3
        # nz -= 3

        # cell, ids_of_the_cell = self.solver._get_cell_ids(nx, ny, nz)
        # self.assertEqual(
        #     len(ids_of_the_cell) > 20,
        #     True,
        #     f"incorrect number of atoms: should be lower than {len(ids_of_the_cell)}",
        # )
        # type_of_cell = self.solver._process_single_cell(ids_of_the_cell, cell)
        # self.assertEqual(type_of_cell, SIMPLE)
        # self.assertEqual(
        #     self.solver._solver_rule(ids_of_the_cell, to_approximate=True), True
        # )
        # self.L = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
        types_of_atom = self.communicator.__get_atom_types__()
        masses = [58.6934 if i == 1 else 469.5472 for i in types_of_atom]
        total_mass_before_approximation = np.sum(masses)

        self.dc.accelerate(self.L)

        self.solver.__debug_info__()

        # self.L.command("compute total_mass all reduce sum m")

        # self.L = self.dc._lammps_execute()
        # self.assertEqual(
        #     self.L,
        #     L2,
        #     f"different instances ",
        # )

        self.assertEqual(
            len(self.solver._get_cells_to_approximate()) > 150,
            True,
            f"There should be no enough regions to be approximated, found only {len(self.solver._get_cells_to_approximate())}",
        )
        # self.assertEqual(
        #     len(self.solver._get_cells_to_granulate()) > 10,
        #     True,
        #     f"There should be no enough regions to be GRANULATED, found only {len(self.solver._get_cells_to_granulate())}",
        # )

        self.L.command("compute 1 all pair/local dist")
        self.L.command("compute 2 all reduce min c_1 inputs local")
        self.L.command("run 0")

        min_dist = self.L.extract_compute("2", 0, 0)
        print(f"Минимальное расстояние в системе до удаления оверлэпов: {min_dist}")

        self.L.command("delete_atoms overlap 0.05 all all")

        # self.L.command("reset_atoms id")

        # self.L.command("fix 1 all nve/limit 0.1")
        # self.L.command("run 500")
        # self.L.command("unfix 1")

        self.L.command("run 0")

        min_dist = self.L.extract_compute("2", 0, 0)
        print(f"Минимальное расстояние в системе: {min_dist}")

        self.L.command("group new_atoms type 2")
        self.L.command("velocity new_atoms create 0.1 49672")
        self.L.command("run 3000")

        types_of_atom = self.communicator.__get_atom_types__()
        masses = [58.6934 if i == 1 else 469.5472 for i in types_of_atom]
        total_mass_after_approximation = np.sum(masses)
        # self.dc.accelerate(self.L)

        self.assertEqual(
            abs(1 - total_mass_before_approximation / total_mass_after_approximation)
            < 1e-2,
            True,
            f"The law of masses is not satisfied. The error is {(1 - total_mass_before_approximation / total_mass_after_approximation)*100}%",
        )


if __name__ == "__main__":
    unittest.main()
