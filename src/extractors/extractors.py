from tarfile import data_filter
from typing import override

import numpy as np
import logging
from ase import Atoms
from ase.io import read
from ase import units
import os
from ase.io import write
from sklearn.cluster import MeanShift
from scipy.ndimage import center_of_mass
from ovito.io.lammps import lammps_to_ovito
from ovito.modifiers import CommonNeighborAnalysisModifier
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from k_means_constrained import KMeansConstrained
from ase.build import bulk
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

from src.extractors.base_extractor import Extractor
from ..utils.utils import LammpsCommunicator

SIMPLE = 0
CRACK = 1
ROGUE_CELL = 2
GRAINED = 3
logger = logging.getLogger(__name__)

plt.close()


class ExampleLayerExtractor(Extractor):
    def __init__(self):
        super().__init__()

    def check_condition_of_region(self, velocities, masses, threshold=20):

        v_group = velocities
        m_group = masses

        total_mass = np.sum(m_group)
        v_com = np.sum(v_group * m_group[:, None], axis=0) / total_mass

        # Relative velocity (internal motion only)
        v_rel = v_group - v_com

        # Kinetic Energy = 0.5 * m * v^2
        AMU_A2PS2_TO_EV = 1.0364269e-4
        n_atoms = v_group.shape[0]

        ke = 0.5 * np.sum(m_group * np.sum(v_rel**2, axis=1)) * AMU_A2PS2_TO_EV

        current_T = 2 * ke / (3 * n_atoms * units.kB)

        if n_atoms > 0:
            current_T = 2 * ke / (3 * n_atoms * units.kB)
            layer_temp = current_T
        else:
            layer_temp = 0.0

        logging.info(
            f"Количество атомов в слое: {len(m_group)}, температура слоя: {layer_temp}"
        )

        if layer_temp > threshold:
            return True

        return False

    def visualize_interesting_regions(
        self,
        coordinates,
        velocities,
        masses,
        lattice_constant,
        lattice_constant_cg,
        criteria="temp",
    ):

        z = coordinates[:, 2]
        step = lattice_constant * 2 + 1e-1
        logging.info(f"Using FCC layer spacing: step = {step}")

        zmax = 40.08
        zmin = z.min()

        layers = []
        logging.error(f"================vvvvvvvvvvvvvvvvvvv===========================")

        for i in range(int((zmax - zmin) // step) + 1):
            upper = zmax - i * step
            lower = upper - step

            if lower < zmin:
                lower = zmin

            logging.info(f"LAYER {i}, Collecting atoms from: {lower} to {upper}")

            mask = (z >= lower) & (z <= upper)
            actual_atoms = coordinates[mask]

            if np.any(masses[mask] > 200):
                logging.info(
                    f"SKIPPED ALREADY GRAINED ATOMS: min: {masses.min()}, max: {masses.max()}"
                )
                continue

            if not self.check_condition_of_region(
                velocities[mask], masses[mask], threshold=10
            ):
                continue

            au = bulk("Au", "fcc", a=lattice_constant_cg, cubic=True)
            target_atoms = len(actual_atoms) / (4**3)

            target_cells = target_atoms / 4.0
            n = int(round(target_cells))

            nx = int(round(n**0.5))
            if nx < 1:
                nx = 1
            ny = int(round(n / nx))
            if ny < 1:
                ny = 1

            au = bulk("Au", "fcc", a=lattice_constant_cg, cubic=True)
            plane = au.repeat((3, 4, 1))
            positions_of_grained = plane.get_positions()
            mask_bebe = positions_of_grained[:, 2] < lattice_constant_cg / 2

            plane = plane[mask_bebe]
            positions_of_grained = plane.get_positions()
            positions_of_grained[:, 2] += (lower + upper) / 2

            layers.append((mask, positions_of_grained))
        return layers

    # def visualize_interesting_regions(self, positions, mode='test'):
    #     pass
    # write("region_mask_already_grained.xyz", atoms[region_mask_already_grained])


class FccCellsExtractor(Extractor):
    def __init__(
        self,
        lammps_extractor: LammpsCommunicator,
        lattice_contant: float,
        lattice_constant_cg=7.04,
        lower_threshold=-3.0,
        upper_threshold=-0.7,
    ):
        super().__init__()

        from ase.build import bulk

        ni = bulk("Ni", "fcc", a=3.52, cubic=True)

        self.model_fcc_positions = ni.repeat((2, 2, 2)).get_positions()
        self.lammps_extractor = lammps_extractor

        ni = bulk("Ni", "fcc", a=3.52 * 2, cubic=True)
        self.model_mega_fcc_positions = ni.repeat((1, 1, 1)).get_positions()
        self.lattice_constant = lattice_contant

        self.cell_size = (
            self.lattice_constant * 2.0
        )  # Accept a) Fluctuations b) Intersections?

        self.LOWER_THRESHOLD = lower_threshold
        self.UPPER_THRESHOLD = upper_threshold

        self.lattice_constant_cg = lattice_constant_cg

        self.cells_to_approximate = []
        self.rogue_cells = []
        self.extra_atoms = []
        self.cells_to_granulate = []
        self.debug_cells_grained = []

    def clear_extractor(self):
        self.cells_to_approximate = []
        self.rogue_cells = []
        self.extra_atoms = []
        self.cells_to_granulate = []
        self.debug_cells_grained = []

    def set_communicator(self, lammps_instance):
        self.lammps_extractor = LammpsCommunicator(lammps_instance)

    def get_communicator(self) -> LammpsCommunicator:
        return self.lammps_extractor

    @override
    def extract_interesting_regions(
        self,
    ):
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi, self.pbc = (
            self.lammps_extractor.__get_box_size__()
        )
        
        # Align the grid to the lattice to prevent thermal fluctuations from shifting the cells
        # self.xlo = self.xlo  np.round(self.xlo / self.lattice_constant) * self.lattice_constant
        # self.ylo =  np.round(self.ylo / self.lattice_constant) * self.lattice_constant
        # self.zlo = np.round(self.zlo / self.lattice_constant) * self.lattice_constant

        self.clear_extractor()
        cell_size = self.lattice_constant * 2.0

        nx = int(np.round((self.xhi - self.xlo) / cell_size))
        ny = int(np.round((self.yhi - self.ylo) / cell_size))
        nz = int(np.round((self.zhi - self.zlo) / cell_size))

        # Split space to cubes with edge cell_size * SCALE_FACTOR
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cell, ids_of_the_cell = self._get_cell_ids(ix, iy, iz)
                    self._process_single_cell(ids_of_the_cell, cell)

        self._repair_underfilled_cells()

    def _get_cell_ids(self, ix, iy, iz):
        """
        Receives the identificators of atoms inside the mega cell
        """
        positions = (
            self.lammps_extractor.__get_positions__()
        )  # Should be sorted by id or not?

        x_exact_min = self.xlo + ix * self.cell_size
        x_exact_max = x_exact_min + self.cell_size
        y_exact_min = self.ylo + iy * self.cell_size
        y_exact_max = y_exact_min + self.cell_size
        z_exact_min = self.zlo + iz * self.cell_size
        z_exact_max = z_exact_min + self.cell_size

        mask = ()
        low, high = 0.0, 12
        for _ in range(40):
            mid = (low + high) / 2

            x_min = x_exact_min - 0.1
            x_max = x_exact_max + 0.1
            y_min = y_exact_min - 0.1 * mid
            y_max = y_exact_max + 0.1 * mid
            z_min = z_exact_min 
            z_max = z_exact_max + 0.1

            # collects atoms in the cell
            mask = (
                (positions[:, 0] >= x_min) &
                (positions[:, 0] <= x_max) &
                (positions[:, 1] >= y_min) &
                (positions[:, 1] <= y_max) &
                (positions[:, 2] >= z_min) &
                (positions[:, 2] <= z_max)
            )
            if np.sum(mask) >= 14:
                high = mid 
            else:
                low = mid

        identificators = self.lammps_extractor.__get_atom_identificators__()[mask]

        return (x_min, x_max, y_min, y_max, z_min, z_max), identificators

    def _process_single_cell(self, atom_identificators, cell):
        """
        Processes each mega cell. There are four possible cases:
        1) The cell is overfilled with atoms (>32 atoms) -> add to dictionary with overfilled cells
        2) The cell is undefilled with atoms (<32 atoms) -> add to dictionary with vacant cells
        3) The cell is already grained (4 atoms)
        4) The cell is in a position of a crack (skip this area) -> skip it
        """
        type_of_cell = SIMPLE
        number_of_atoms_in_cell = len(atom_identificators)
        if number_of_atoms_in_cell > 30:

            if self._solver_rule(atom_identificators, to_approximate=True):
                self.cells_to_approximate.append((cell, atom_identificators))
        elif number_of_atoms_in_cell > 10 and number_of_atoms_in_cell < 15:
            type_of_cell = GRAINED
            if self._solver_rule(atom_identificators, to_granulate=True):
                self.cells_to_granulate.append((cell, atom_identificators))
        # elif number_of_atoms_in_cell > 32:
        #     new_atom_identificators = self._extract_extra_atoms(atom_identificators, is_crack=False)
        #     if self._solver_rule(atom_identificators, to_approximate=True):
        #         self.cells_to_approximate.append((cell, new_atom_identificators))
        # elif number_of_atoms_in_cell < 32:
        #     type_of_cell = self._define_type_of_underfilled_cell(atom_identificators)
        #     if type_of_cell == CRACK:
        #         self._extract_extra_atoms(atom_identificators, is_crack=True)
        #     elif type_of_cell == ROGUE_CELL:
        #         if self._solver_rule(atom_identificators, to_granulate=True):
        #             self.cells_to_granulate.append((cell, atom_identificators)) # TODO: DELETE!!!
        #         self.rogue_cells.append((cell, atom_identificators))
        #     elif type_of_cell == GRAINED:
        #         if self._solver_rule(atom_identificators, to_granulate=True):
        #             self.cells_to_granulate.append((cell, atom_identificators))
        #     else:
        #         if self._solver_rule(atom_identificators, to_approximate=True):
        #             self.cells_to_approximate.append((cell, atom_identificators))

        return type_of_cell

    def get_lammps_instance(self):
        return self.lammps_extractor.get_instance()

    def _solver_rule(
        self, atom_identificators, to_approximate=False, to_granulate=False
    ) -> bool:
        res = False
        mean_pe = np.mean(
            self.lammps_extractor.__get_pe_per_atom__()[atom_identificators]
        )
        minima = np.min(
            self.lammps_extractor.__get_pe_per_atom__()[atom_identificators]
        )
        print(f"Min PE: {minima}")
        maxima = np.max(
            self.lammps_extractor.__get_pe_per_atom__()[atom_identificators]
        )
        print(f"Max PE: {maxima}")
        # print(f"Mean PE: {np.mean(self.lammps_extractor.__get_pe_per_atom__()[atom_identificators])}")
        if to_approximate:
            if mean_pe < self.LOWER_THRESHOLD:
                res = True
        if to_granulate:
            if mean_pe > self.UPPER_THRESHOLD:
                res = True
        return res

    def _define_type_of_underfilled_cell(self, atom_identificators) -> int:
        res = SIMPLE
        atom_types = self.lammps_extractor.__get_atom_types__()[atom_identificators]
        number_of_small_atoms = len(atom_types[atom_types == 1])
        number_of_huge_atoms = len(atom_types[atom_types == 2])

        # TODO: correct defining of a crack!!! NOW IT IS INCORRECT!!!
        if number_of_small_atoms > 20:
            res = ROGUE_CELL
        elif number_of_huge_atoms == 4:
            res = GRAINED
        return res

    def _lammps_execute(self):
        return self.get_communicator().get_instance()

    def _execute_lammps_replacement_approximation(self, cell_to_granulate: tuple):
        """
        Replace atoms with new one.
        """
        (x_min, x_max, y_min, y_max, z_min, z_max), atom_ids = cell_to_granulate
        
        if len(atom_ids) > 0:
            string_of_ids = " ".join(map(str, atom_ids + 1))
            self._lammps_execute().command(f"group delete_group id {string_of_ids}")
            self._lammps_execute().command("delete_atoms group delete_group")
            self._lammps_execute().command("group delete_group delete")

        self._lammps_execute().command(
            f"region spawn_region block {x_min} {x_max} {y_min} {y_max} {z_min} {z_max} units box"
        )

        lenj = len(self.get_communicator().__get_atom_identificators__())
        print(f"Max len: {lenj}")
        self._lammps_execute().command(f"lattice fcc {self.lattice_constant_cg-0.01}")
        
        if len(atom_ids) > 0:
            velocities_of_the_cell = self.get_communicator().__get_velocities__()[atom_ids]
            mean_vx = np.mean(velocities_of_the_cell[:, 0]) * 8
            mean_vy = np.mean(velocities_of_the_cell[:, 1]) * 8
            mean_vz = np.mean(velocities_of_the_cell[:, 2]) * 8
        else:
            mean_vx = mean_vy = mean_vz = 0.0

        commands = [
            f"variable vx_new equal {mean_vx}",
            f"variable vy_new equal {mean_vy}",
            f"variable vz_new equal {mean_vz}",
            "create_atoms 2 region spawn_region",
            "group cell_atoms region spawn_region",
            "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
            "variable vx_new delete",
            "variable vy_new delete",
            "variable vz_new delete",
        ]
        for cmd in commands:
            self._lammps_execute().command(cmd)
        self._lammps_execute().command("group cell_atoms delete")
        self._lammps_execute().command("region spawn_region delete")
        self._lammps_execute().command("reset_atoms id")
        # self._lammps_execute().command("delete_atoms overlap 0.01 all all")
        # return
        # if self.__DEBUG_MODE__:
        #     self.__debug_grained_cells.append((x_min, x_max, y_min, y_max, z_min, z_max))

    def _execute_lammps_replacement_granulation(self, cell_to_granulate: tuple):
        (x_min, x_max, y_min, y_max, z_min, z_max), atom_ids = cell_to_granulate
        
        if len(atom_ids) > 0:
            string_of_ids = " ".join(map(str, atom_ids + 1))
            self._lammps_execute().command(f"group delete_group id {string_of_ids}")
            self._lammps_execute().command("delete_atoms group delete_group")
            self._lammps_execute().command("group delete_group delete")

        self._lammps_execute().command(
            f"region spawn_region block {x_min} {x_max} {y_min} {y_max} {z_min} {z_max} units box"
        )

        self._lammps_execute().command(f"lattice fcc {self.lattice_constant}")
        
        if len(atom_ids) > 0:
            velocities_of_the_cell = self.get_communicator().__get_velocities__()[atom_ids]
            mean_vx = np.mean(velocities_of_the_cell[:, 0]) / 8
            mean_vy = np.mean(velocities_of_the_cell[:, 1]) / 8
            mean_vz = np.mean(velocities_of_the_cell[:, 2]) / 8
        else:
            mean_vx = mean_vy = mean_vz = 0.0

        commands = [
            f"variable vx_new equal {mean_vx}",
            f"variable vy_new equal {mean_vy}",
            f"variable vz_new equal {mean_vz}",
            "create_atoms 1 region spawn_region",
            "group cell_atoms region spawn_region",
            "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
            "variable vx_new delete",
            "variable vy_new delete",
            "variable vz_new delete",
        ]
        for cmd in commands:
            self._lammps_execute().command(cmd)
        self._lammps_execute().command("group cell_atoms delete")
        self._lammps_execute().command("region spawn_region delete")
        # self._lammps_execute().command("delete_atoms overlap 0.01 all all")
        # return
        # if self.__DEBUG_MODE__:
        #     self.__debug_grained_cells.append((x_min, x_max, y_min, y_max, z_min, z_max))

    def _extract_extra_atoms(self, atom_identificators: np.ndarray, is_crack: bool):
        if is_crack:
            self.extra_atoms.extend(atom_identificators)
        else:
            # Extract those atoms which are far from ideal fcc cell
            self.extra_atoms.extend(atom_identificators[32:])

            # TODO: correct fcc extraction. NOW IT IS INCORRECT!!!
            return atom_identificators[:32]

    def _repair_underfilled_cells(
        self,
    ):

        # TODO: setup satisfaction of law of masses!!!
        # Fill unfilled mega cells.
        return

    def _get_cells_to_approximate(
        self,
    ) -> list:
        return self.cells_to_approximate

    def _get_cells_to_granulate(
        self,
    ) -> list:
        return self.cells_to_granulate
