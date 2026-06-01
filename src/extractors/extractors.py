import enum
import types
from typing import override

import numpy as np
import logging
from ase import Atoms
from ase.io import read
from ase import units
import os
from ase.io import write
from scipy.spatial import cKDTree
from ase.build import bulk
import matplotlib.pyplot as plt
import numpy as np

from src.extractors.base_extractor import Extractor
from ..utils.utils import LammpsCommunicator

SIMPLE = 0
CRACK = 1
ROGUE_CELL = 2
GRAINED = 3

NUMBER_OF_ATOMS_IN_FCC_CELL = 4
PHANTOM_LAYER_SIZE = 2

APPROXIMATE = -1897398
GRANULATE = 23920932
INFERENCE = 999
RANDOM_CONDITION = 28383829
logger = logging.getLogger(__name__)

plt.close()

from enum import Enum
import random

class Condition(Enum):
    RAND = 1
    POTENTIAL_ENERGY = 2


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
        scale_factor : int,
        lattice_constant_cg=7.04,
        lower_threshold=-2.5, # approximates
        upper_threshold=-0.7, # granulates
        smoke_test = INFERENCE,
    ):
        super().__init__()

        from ase.build import bulk

        ni = bulk("Ni", "fcc", a=3.52, cubic=True)
        self.scale_factor = scale_factor

        self.model_fcc_positions = ni.repeat((2, 2, 2)).get_positions()
        self.lammps_extractor = lammps_extractor

        ni = bulk("Ni", "fcc", a=3.52 * 2, cubic=True)
        self.model_mega_fcc_positions = ni.repeat((1, 1, 1)).get_positions()
        self.lattice_constant = lattice_contant
        a = lattice_contant

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

        self.basis = np.array([
            [a/2, 0, 0],
            [0, a/2, 0],
            [0, 0, a/2]
        ])

        self.template_of_filling = np.array([[0, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [2, 0, 0],
        [3, 1, 0],
        [3, 0, 1],
        [2, 1, 1],
        [0, 2, 0],
        [1, 3, 0],
        [1, 2, 1],
        [0, 3, 1],
        [2, 2, 0],
        [3, 3, 0],
        [3, 2, 1],
        [2, 3, 1],
        [0, 0, 2],
        [1, 1, 2],
        [1, 0, 3],
        [0, 1, 3],
        [2, 0, 2],
        [3, 1, 2],
        [3, 0, 3],
        [2, 1, 3],
        [0, 2, 2],
        [1, 3, 2],
        [1, 2, 3],
        [0, 3, 3],
        [2, 2, 2],
        [3, 3, 2],
        [3, 2, 3],
        [2, 3, 3]])

        self.smoke_test = smoke_test

        self.z_group_decomposition = None

        self.phantom_part = []

    def __debug_info__(self,):
        atom_counts = [item[0] for item in self.debug_cells_grained]
        pe_values = [item[1] for item in self.debug_cells_grained]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(atom_counts, bins=20, edgecolor='black', alpha=0.8)
        axes[0].axvline(x=32, color='red', linestyle='--', label='Ideal FCC (32 atoms)')
        axes[0].axvline(x=28, color='orange', linestyle=':', label='Lower bound (28)')
        axes[0].axvline(x=36, color='orange', linestyle=':', label='Upper bound (36)')
        axes[0].set_xlabel('Number of atoms in cell')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Atom count distribution')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.5)
        
        # Гистограмма 2: Распределение по средней потенциальной энергии
        axes[1].hist(pe_values, bins=30, edgecolor='black', alpha=0.8, color='steelblue')
        axes[1].axvline(x=self.LOWER_THRESHOLD, color='green', linestyle='--', 
                    label=f'LOWER_THRESHOLD ({self.LOWER_THRESHOLD})')
        axes[1].axvline(x=self.UPPER_THRESHOLD, color='red', linestyle='--', 
                    label=f'UPPER_THRESHOLD ({self.UPPER_THRESHOLD})')
        axes[1].set_xlabel('Mean potential energy per atom')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Mean PE distribution')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('debug_cell_distribution.png', dpi=150)

        print(f"\n📊 Статистика по {len(atom_counts)} ячейкам:")
        print(f"Атомы: min={min(atom_counts)}, max={max(atom_counts)}, mean={np.mean(atom_counts):.1f}, std={np.std(atom_counts):.1f}")
        print(f"PE: min={min(pe_values):.3f}, max={max(pe_values):.3f}, mean={np.mean(pe_values):.3f}, std={np.std(pe_values):.3f}")
        
        # Сколько ячеек проходит твои текущие фильтры?
        passed_atoms = sum(1 for n in atom_counts if 28 <= n <= 36)
        passed_pe_approx = sum(1 for pe in pe_values if pe < self.LOWER_THRESHOLD)
        passed_pe_gran = sum(1 for pe in pe_values if pe > self.UPPER_THRESHOLD)
        
        print(f"\n🎯 Проходят фильтры:")
        print(f"По атомам (28-36): {passed_atoms}/{len(atom_counts)} ({100*passed_atoms/len(atom_counts):.1f}%)")
        print(f"PE < LOWER ({self.LOWER_THRESHOLD}): {passed_pe_approx}/{len(pe_values)} ({100*passed_pe_approx/len(pe_values):.1f}%)")
        print(f"PE > UPPER ({self.UPPER_THRESHOLD}): {passed_pe_gran}/{len(pe_values)} ({100*passed_pe_gran/len(pe_values):.1f}%)")

        return self.debug_cells_grained

    def clear_extractor(self):
        self.fcc_cell_to_approximate = []
        self.fcc_cell_to_granulate = []

    def set_communicator(self, lammps_instance):
        self.lammps_extractor = LammpsCommunicator(lammps_instance)

    def get_communicator(self) -> LammpsCommunicator:
        return self.lammps_extractor

    def get_basis_decompostion(self,):
        positions = self.lammps_extractor.__get_positions__()

        coeffs = positions @ np.linalg.inv(self.basis.T)

        coeffs = np.round(coeffs).astype(int)

        self.z_group_decomposition = coeffs
        return self.z_group_decomposition

    def get_group_fcc_cells(self, scale_factor :int):

        if self.z_group_decomposition is None:
            self.z_group_decomposition = self.get_basis_decompostion()

        # These are atoms to be approximated with larger particles
        self.first_level_mask = np.all(self.z_group_decomposition % scale_factor == 0, axis=1) # & (np.sum(coeffs, axis=1) % 3 == 0)
        self.first_level_mask_basis = np.sum(self.z_group_decomposition,axis=1) % (scale_factor*2) == 0
        # assert np.sum(self.first_level_mask) % len(coeffs) // (2*scale_factor)**3, f"What the FUCK? Found {len(self.z_group_symmetry[self.first_level_mask])} instead of expected {len(coeffs) // scale_factor**3}. Fix this shit NOW!"
        # These are atoms to be removed! 
        self.second_level_mask = ~self.first_level_mask

        # one mega fcc cell
        
        # assert len(self.template_of_filling) == 32 
 
        # Generate masks of fcc megacells, size of scale_factor
        # coeffs_atoms = coeffs[atom_types == 1]
        # coeffs_particles = coeffs[atom_types == 2]

        if scale_factor == 1:
            number_of_cells = self.z_group_decomposition[:, 1].max() // 2
        else:
            number_of_cells = (self.z_group_decomposition[:, 1].max()) // (1 + scale_factor)
        
        # if scale_factor != 1:
        step = (scale_factor*2) ## ADD BUFFER. SO EVERY BUFFER IS A PHANTOM
        scaled = (self.z_group_decomposition / (1 + scale_factor))
        scaled = self.z_group_decomposition
        # else:
        #     step = 1
        #     scaled = self.z_group_decomposition

        n_cells = int(number_of_cells)
        masks = []

        max_y = self.z_group_decomposition[:, 1].max()

        for x in range(0, max_y, step):
            for y in range(0, max_y, step):
                for z in range(0, max_y, step):
                    mask = (
                        (scaled[:, 0] >= x) & (scaled[:, 0] <= x + (step-1)) &
                        (scaled[:, 1] >= y) & (scaled[:, 1] <= y + (step-1)) &
                        (scaled[:, 2] >= z) & (scaled[:, 2] <= z + (step-1))
                    )

                    # if (positions[mask] == 32) or (positions[mask] == 29) or \
                    # (positions[mask] == 32) or (positions[mask] == 29) or \
                    masks.append(mask)

        self.fcc_mega_cells = masks
        return self.fcc_mega_cells
    
    def __safe_unique_ids__(self, ids_to_delete):
        if not ids_to_delete or len(ids_to_delete) == 0:
            return np.array([], dtype=int)
        return np.unique(np.concatenate(ids_to_delete).flatten()).astype(int)
    
    def _check_overlapping_atoms(self, positions_to_spawn, positions, min_distance=0.1):
        if len(positions) == 0:
            return True
    
        tree = cKDTree(positions)
        distances, _ = tree.query(positions_to_spawn, k=1)
        
        return np.all(distances >= min_distance)
    
    def _add_phantom_layer_of_atoms(self, fcc_mega_cells):
        mask_all_mega_cells = np.any(fcc_mega_cells, axis=1)

        mask_all_mega_cells_with_phantom = mask_all_mega_cells
        
        offsets = np.array(np.meshgrid(
            range(-PHANTOM_LAYER_SIZE, PHANTOM_LAYER_SIZE+1),
            range(-PHANTOM_LAYER_SIZE, PHANTOM_LAYER_SIZE+1),
            range(-PHANTOM_LAYER_SIZE, PHANTOM_LAYER_SIZE+1)
        )).T.reshape(-1, 3)

        expanded_coords = []
        for coord in mask_all_mega_cells_with_phantom:
            expanded_coords.append(coord + offsets)

        all_coords = np.vstack(expanded_coords)
        mask_all_mega_cells_with_phantom = np.unique(all_coords, axis=0)

        phantom_positions = ~(mask_all_mega_cells_with_phantom & mask_all_mega_cells)

        return phantom_positions

    def get_cells_to_apply_action(self):
        self.clear_extractor()

        self.get_group_fcc_cells(scale_factor=self.scale_factor)
        identificators = self.lammps_extractor.__get_atom_identificators__()
        atom_types = self.lammps_extractor.__get_atom_types__()
        pe_per_atom = self.lammps_extractor.__get_pe_per_atom__()

        self.fcc_cell_to_approximate = []
        self.fcc_cell_to_granulate = []

        for fcc_cell in self.fcc_mega_cells:  # fcc_cell is a boolean mask of shape (N,)
            # if ((len(atom_types[fcc_cell]) != NUMBER_OF_ATOMS_IN_FCC_CELL*self.scale_factor**3) or (len(atom_types[fcc_cell]) == NUMBER_OF_ATOMS_IN_FCC_CELL)):
            #     continue
            atoms_in_cell = pe_per_atom[fcc_cell]
            types_in_cell = atom_types[fcc_cell]
            if (len(pe_per_atom[fcc_cell]) == 0) or len(pe_per_atom[fcc_cell]) % 4 != 0:
                continue

            mean_pe = np.mean(atoms_in_cell)

            if self.smoke_test == APPROXIMATE:
                self.fcc_cell_to_approximate.append(fcc_cell)
                # break

            if self.smoke_test == GRANULATE:
                self.fcc_cell_to_granulate.append(fcc_cell)
            
            if self.smoke_test == RANDOM_CONDITION:
                if random.random() > 0.8 and np.all(types_in_cell == 1):
                    self.fcc_cell_to_approximate.append(fcc_cell)
                elif random.random() > 0.5 and np.all(types_in_cell == 2):
                    self.fcc_cell_to_granulate.append(fcc_cell)

            if self.smoke_test == INFERENCE:
                if np.all(atoms_in_cell < self.LOWER_THRESHOLD) and np.all(types_in_cell == 1):
                    self.fcc_cell_to_approximate.append(fcc_cell)
                elif np.all(atoms_in_cell > self.UPPER_THRESHOLD) and np.all(types_in_cell == 2):
                    self.fcc_cell_to_granulate.append(fcc_cell)     
    
    def get_data_of_cells_to_approximate(self, add_phantom_layer=True):
        positions = self.lammps_extractor.__get_positions__()
        identificators = self.lammps_extractor.__get_atom_identificators__()
        ids_tochange_with = []
        ids_to_delete = []
        positions_of_large = []
        phantom_atoms = []

        for cell in self.fcc_cell_to_approximate:
            # if len(identificators[cell & self.first_level_mask]) != 8:
            #     continue
            # assert np.sum(cell & self.first_level_mask) == NUMBER_OF_ATOMS_IN_FCC_CELL
            # ids_tochange_with.append(identificators[(cell & self.first_level_mask)])
            # ids_tochange_with.append(np.where((cell & self.first_level_mask))[0]+1)
            positions_of_large.append(cell & self.first_level_mask & self.first_level_mask_basis)
            # assert np.sum(cell & self.second_level_mask) == 56 == len(identificators[cell & self.second_level_mask])
            # ids_to_delete.append(identificators[cell & self.second_level_mask])
            ids_to_delete.append(np.where(cell)[0]+1)


        if add_phantom_layer:
            phantom_atoms = self._add_phantom_layer_of_atoms(self.fcc_cell_to_approximate)
        ids_atoms_to_change_with_particles = np.array(ids_tochange_with).flatten()
        mask = np.any(np.array(positions_of_large), axis=0)

        positions_of_large_and_phantom = np.concatenate([positions[positions_of_large], positions[phantom_atoms]+ 0.01])

        return self.__safe_unique_ids__(ids_to_delete), phantom_atoms, positions_of_large_and_phantom

    def get_data_of_cells_to_granulate(self, add_phantom_layer=True):
        identificators = self.lammps_extractor.__get_atom_identificators__()
        positions = self.lammps_extractor.__get_positions__()

        if self.z_group_decomposition is None:
            self.z_group_decomposition = self.get_basis_decompostion()

        ids_tochange_with = []
        all_positions_to_spawn = np.array([])
        phantom_atoms = []

        if add_phantom_layer:
            phantom_atoms = self._add_phantom_layer_of_atoms(self.fcc_cell_to_approximate)

        for cell in self.fcc_cell_to_granulate:
            # assert np.sum(cell & self.first_level_mask) == 4 == len(identificators[cell & self.first_level_mask])
            ids_tochange_with.append(np.where(cell)[0]+1)

            idx = np.argmin(self.z_group_decomposition[cell][:, 0]) #cell & self.first_level_mask

            min_point = self.z_group_decomposition[cell][idx] #cell & self.first_level_mask

            positions_to_spawn = (self.basis @ (self.template_of_filling + min_point).T).T

            # if not self._check_overlapping_atoms(positions_to_spawn, positions[~cell]):
            #     continue

            if len(all_positions_to_spawn) == 0:
                all_positions_to_spawn = positions_to_spawn
            else:
                all_positions_to_spawn = np.concatenate([all_positions_to_spawn, positions_to_spawn])

        # assert len(positions_to_spawn) % 32 == 0, f'NO! The number of positions is {len(positions_to_spawn)}!' 
        assert len(self.__safe_unique_ids__(ids_tochange_with)) % 4 ==0, f'FUCK: the number of units to be deleted is {len(self.__safe_unique_ids__(ids_tochange_with))}, and spawn is {len(positions_to_spawn)}'
        return self.__safe_unique_ids__(ids_tochange_with), all_positions_to_spawn, phantom_atoms

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
        positions = self.lammps_extractor.__get_positions__()
        atom_ids = self.lammps_extractor.__get_atom_identificators__()

        cell_size = self.lattice_constant * 2.0  # 7.04
        gap = self.lattice_constant - 2*1e-1  # зазор между боксами (Å)
        stride = cell_size + gap
        cell_size_with_eps = cell_size+1e-2

        box = self.get_communicator().get_instance().extract_box()
        xlo, xhi = box[0][0], box[1][0]
        ylo, yhi = box[0][1], box[1][1]
        zlo, zhi = box[0][2], box[1][2]
        eps = 1e-1

        nx = int(np.ceil((xhi - xlo) / cell_size))
        ny = int(np.ceil((yhi - ylo) / cell_size))
        nz = int(np.ceil((zhi - zlo) / cell_size))

        for ix in range(nx):
            x_min = xlo + ix * cell_size
            x_max = min(xlo + (ix+1) * cell_size, xhi)   # последняя ячейка может быть меньше
            for iy in range(ny):
                y_min = ylo + iy * cell_size
                y_max = min(ylo + (iy+1) * cell_size, yhi)
                for iz in range(nz):
                    z_min = zlo + iz * cell_size
                    z_max = min(zlo + (iz+1) * cell_size, zhi)

                    mask = (positions[:,0] >= x_min - eps) & (positions[:,0] < x_max + eps) & \
                        (positions[:,1] >= y_min - eps) & (positions[:,1] < y_max + eps) & \
                        (positions[:,2] >= z_min - eps) & (positions[:,2] < z_max + eps)
                    cell_ids = atom_ids[mask]
                    if len(cell_ids) == 0:
                        continue
                    cell = (x_min, x_max, y_min, y_max, z_min, z_max)
                    self._process_single_cell(cell_ids, cell)


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

        if number_of_atoms_in_cell > 20 and number_of_atoms_in_cell  < 40:
            if self._solver_rule(atom_identificators, to_approximate=True):
                self.cells_to_approximate.append((cell, atom_identificators))

        if number_of_atoms_in_cell == 4:
            if self._solver_rule(atom_identificators, to_granulate=True):
                self.cells_to_granulate.append((cell, atom_identificators))


        #     if self._solver_rule(atom_identificators, to_approximate=True):
        #         self.cells_to_approximate.append((cell, atom_identificators))
        # elif number_of_atoms_in_cell > 10 and number_of_atoms_in_cell < 15:
        #     type_of_cell = GRAINED
        #     if self._solver_rule(atom_identificators, to_granulate=True):
        #         self.cells_to_granulate.append((cell, atom_identificators))
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

    def _get_cells_to_approximate(
        self,
    ) -> list:
        return self.cells_to_approximate

    def _get_cells_to_granulate(
        self,
    ) -> list:
        return self.cells_to_granulate
