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


class TestFccCellsExtraction2x2(unittest.TestCase):

    def setUp(self):
        A = 3.52
        A_CG = A*2

        self.NUMBER_OF_MEGA_CELLS=2**3

        self.L = lammps()

        self.L.file("tests/lammps_scripts/test_2_cube.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, smoke_test=APPROXIMATE, scale_factor=2)
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_simple_split_many_cells(self):
        fcc_cells = self.solver.get_group_fcc_cells(scale_factor=1)

        self.assertEqual(4**3, len(fcc_cells))

        for cell in self.solver.fcc_mega_cells:
            self.assertEqual(len(self.communicator.__get_positions__()[cell]), 4)

    def test_simple_split_scale_factor2(self):

        fcc_cells = self.solver.get_group_fcc_cells(scale_factor=2)
 
        self.assertEqual(2**3, len(fcc_cells))

        for cell in fcc_cells:
            self.assertEqual(len(self.communicator.__get_positions__()[cell]), 32)

    def test_approximate_all_simulation(self):
        """
        expect changer:
         1) to approximate 1 mega fcc cell (2*2*2) with 4 atoms
         2) to put these atoms in correct positions
        with respect to law of masses and potential energy!
        """

        initial_mass = self.communicator.get_total_mass()

        self.solver.get_cells_to_apply_action()

        identificators_to_delete, positions_of_large = self.solver.get_data_of_cells_to_approximate()

        self.assertEqual(len(positions_of_large), self.NUMBER_OF_MEGA_CELLS * 4)

        self.assertEqual(len(identificators_to_delete), self.NUMBER_OF_MEGA_CELLS * 32)

        self.dc.accelerate(self.L)

        self.L = self.dc.communicator.get_instance()
        self.communicator2 = LammpsCommunicator(self.L)

        final_mass = self.communicator2.get_total_mass()

        self.L.command("run 1500")

        #TODO: CHECK why mass is not satisfied properly
        self.assertLess(abs(1 - initial_mass / final_mass)*100, 5)
        
    def test_granulate_all_simulation(self):
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

        initial_mass = self.communicator.get_total_mass()

        self.dc.accelerate(self.L)
        positions_old = self.communicator.__get_positions__()
        self.dc.communicator.get_instance().command("run 1500")
        self.L = self.dc.communicator.get_instance()

        self.communicator = LammpsCommunicator(self.L)
        # self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
        #     self.communicator.__get_box_size__()
        # )
        solver = FccCellsExtractor(self.communicator, 3.52, scale_factor=2, smoke_test=GRANULATE)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        dc.accelerate(self.L, only_granulate=True)

        self.communicator2 = LammpsCommunicator(dc.communicator.get_instance())

        final_mass = self.communicator2.get_total_mass()
        self.L.command("run 1500")

        self.assertLess(abs(1 - initial_mass / final_mass)*100, 1, f"Law of masses is not satisfied! The error is {np.round(abs(1 - initial_mass / final_mass)*100)}%. Initial mass is {initial_mass}, final is {final_mass}")

    def test_random_condition_approximation_granulation(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L

        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        for idx_iteration in range(10):
            dc.accelerate(L)
            L = dc.communicator.get_instance()

            L.command("run 1500")
            L.command("reset_atoms id sort yes")
            communicator = LammpsCommunicator(L) 
            solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
            dc = DynamicChanger(
                communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
            )

            current_mass = communicator.get_total_mass()

            #TODO CHECK LAW OF MASSES!!!
            self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied on iteration {idx_iteration}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        return
    
class TestFccCellsExtraction3x3_with_NPT_and_CRACK(unittest.TestCase):

    def setUp(self):
        A = 3.52
        A_CG = A*2

        self.NUMBER_OF_MEGA_CELLS=2**3

        self.L = lammps()

        self.L.file("tests/lammps_scripts/test_3_cube.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, smoke_test=APPROXIMATE, scale_factor=2)
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_random_condition_no_physics(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L

        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=APPROXIMATE)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        for idx_iteration in range(10):
            dc.accelerate(L)
            L = dc.communicator.get_instance()

            L.command("run 1500")
            L.command("reset_atoms id sort yes")
            communicator = LammpsCommunicator(L) 
            solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
            dc = DynamicChanger(
                communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
            )

            current_mass = communicator.get_total_mass()

            self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied on iteration {idx_iteration}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        return
    
    def test_random_condition_with_nve_ensemle(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L

        L.command("fix     rough all nve")

        # L.command("fix     rough all nvt temp 100.0 300.0 100")

        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=APPROXIMATE)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )
        dc.accelerate(L)
        L.command("run 8000")

        # for idx_iteration in range(10):
        #     dc.accelerate(L)
        L = dc.communicator.get_instance()

        #     L.command("run 1500")
        #     L.command("reset_atoms id sort yes")
        communicator = LammpsCommunicator(L) 
        #     solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
        #     dc = DynamicChanger(
        #         communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        #     )

        current_mass = communicator.get_total_mass()

        #     #TODO CHECK LAW OF MASSES!!!
        self.assertLess(abs(1 - initial_mass / current_mass)*100, 5, f"Law of masses is not satisfied on iteration {0}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        # return
    
    def test_random_condition_with_npt_ensemle(self):

        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L

        L.command("fix     rough all nve")

        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        for idx_iteration in range(10):
            dc.accelerate(L)
            L = dc.communicator.get_instance()

            dc.run_with_scaled_potential()

            # L.command("run 1500")
                # L.command("reset_atoms id sort yes")
                # communicator = LammpsCommunicator(L) 
                # solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
                # dc = DynamicChanger(
                #     communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
                # )

                # current_mass = communicator.get_total_mass()

            #TODO CHECK LAW OF MASSES!!!
            # self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied on iteration {idx_iteration}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        return
    
    def test_random_condition_with_npt_ensemle_and_crack(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L

        L.command("fix      relax all npt temp 100 300 0.1")
        L.command("region   crack block 2 4 2 4 0 INF units lattice")
        L.command("delete_atoms region  crack")

        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        for idx_iteration in range(10):
            dc.accelerate(L)
            L = dc.communicator.get_instance()

            L.command("run 1500")
            L.command("reset_atoms id sort yes")
            communicator = LammpsCommunicator(L) 
            solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
            dc = DynamicChanger(
                communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
            )

            current_mass = communicator.get_total_mass()

            #TODO CHECK LAW OF MASSES!!!
            self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied on iteration {idx_iteration}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        return
    

    
class ScaledCrackPropagation(unittest.TestCase):
    def setUp(self):
        A = 3.52
        A_CG = A*2

        self.NUMBER_OF_MEGA_CELLS=2**3

        self.L = lammps()

        self.L.file("tests/lammps_scripts/crack_ni_lg_velocity_set.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, smoke_test=APPROXIMATE, scale_factor=2, lower_threshold=-3.)
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_baseline(self):

        L = self.L

        L.command("run 20000")

    def test_approximate(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L

        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=APPROXIMATE)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )
        dc.accelerate(L)
        L = dc.communicator.get_instance()
        communicator = LammpsCommunicator(L)

        current_mass = communicator.get_total_mass()

        self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")
        L.command("run 10000")

    def test_random_condition(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L

        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        for idx_iteration in range(10):
            dc.accelerate(L)
            L = dc.communicator.get_instance()

            L.command("run 1500")
            L.command("reset_atoms id sort yes")
            communicator = LammpsCommunicator(L) 
            solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=RANDOM_CONDITION)
            dc = DynamicChanger(
                communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
            )

            current_mass = communicator.get_total_mass()

            #TODO CHECK LAW OF MASSES!!!
            self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied on iteration {idx_iteration}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        return
    
    def test_pe_criteria(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L
        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=INFERENCE)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        for idx_iteration in range(10):
            dc.accelerate(L)
            L = dc.communicator.get_instance()

            L.command("run 1500")
            L.command("reset_atoms id sort yes")
            communicator = LammpsCommunicator(L) 
            solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=INFERENCE)
            dc = DynamicChanger(
                communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
            )

            current_mass = communicator.get_total_mass()

            #TODO CHECK LAW OF MASSES!!!
            self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied on iteration {idx_iteration}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        return
    

class ScaledCrackPropagationFixDeform(unittest.TestCase):
    def setUp(self):
        A = 3.52
        A_CG = A*2

        self.NUMBER_OF_MEGA_CELLS=2**3

        self.L = lammps()

        self.L.file("tests/lammps_scripts/crack_ni.in")

        self.communicator = LammpsCommunicator(self.L)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = (
            self.communicator.__get_box_size__()
        )
        self.solver = FccCellsExtractor(self.communicator, A, smoke_test=APPROXIMATE, scale_factor=2, lower_threshold=-3.)
        self.dc = DynamicChanger(
            self.communicator, self.solver, A, A_CG, scale_factor=2, baby_mode=True
        )

    def test_baseline(self):

        L = self.L

        L.command("run 20000")

    def test_pe_criteria(self):
        """
        almost real task:
        1) based on random condition the approximation is applied
        2) then, if exists regions to be granulated and random condition is satisfied - granulation is provided
        3) run N steps
        4) start over again 
        """

        L = self.L
        communicator = LammpsCommunicator(L)

        initial_mass = communicator.get_total_mass()

        solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=INFERENCE)
        dc = DynamicChanger(
            self.communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
        )

        for idx_iteration in range(20):
            dc.accelerate(L)
            L = dc.communicator.get_instance()

            L.command("run 1000")
            L.command("reset_atoms id sort yes")
            communicator = LammpsCommunicator(L) 
            solver = FccCellsExtractor(communicator, 3.52, scale_factor=2, smoke_test=INFERENCE)
            dc = DynamicChanger(
                communicator, solver, 3.52, 3.52*2, scale_factor=2, baby_mode=True
            )

            current_mass = communicator.get_total_mass()

            #TODO CHECK LAW OF MASSES!!!
            self.assertLess(abs(1 - initial_mass / current_mass)*100, 1, f"Law of masses is not satisfied on iteration {idx_iteration}! The error is {np.round(abs(1 - initial_mass / current_mass)*100)}%. Initial mass is {initial_mass}, final is {current_mass}.")

        return