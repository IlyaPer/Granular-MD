APPROXIMATE = 1
GRANULATE = 2
import logging
from re import S
import numpy as np

from src.extractors.extractors import FccCellsExtractor
from src.utils.utils import LammpsCommunicator

TIME_WINDOW=10

class DynamicChanger():
    def __init__(self, communicator : LammpsCommunicator, extractor : FccCellsExtractor, lattice_constant : float, lattice_constant_cg : float, baby_mode=False):
        self.extractor = extractor
        self.communicator = communicator
        self.baby_mode = baby_mode
        self.lattice_constant_cg = lattice_constant_cg
        self.lattice_constant = lattice_constant
        self.__debug_grained_cells = []
        self.__DEBUG_MODE__ = True
        
        self.snapshots_positions = []
        self.iter_number = 0

    def set_lammps(self, lammps_instance):
        self.communicator = lammps_instance
    
    def _lammps_execute(self):
        return self.communicator.get_instance()
    
    def init_cells(self, lammps_instance):
        self.extractor.set_communicator(lammps_instance)
        # self.extractor.get_communicator().set_positions(np.mean(np.array(self.snapshots_positions), axis=0))

        self.extractor.extract_interesting_regions()
        self.tree_ids_atoms = 

    def accelerate(self, lammps_instance):
        """
        affirmative. execute acceleration.
        """
        # if self.iter_number != TIME_WINDOW:

            # self.snapshots_positions.append(self.communicator.__get_positions__())
            # self.iter_number += 1
            # return

        self.iter_number = 0
        self.extractor.set_communicator(lammps_instance)
        # self.extractor.get_communicator().set_positions(np.mean(np.array(self.snapshots_positions), axis=0))

        self.extractor.extract_interesting_regions()

        cells = self.extractor
        
        for cell_to_granulate in self.extractor._get_cells_to_granulate():
            self._execute_lammps_replacement_granulation(cell_to_granulate)

        # if not self.baby_mode:
        # for cell_to_approximate in self.extractor._get_cells_to_approximate():
        #     # break
        #     self._execute_lammps_replacement_approximation(cell_to_approximate)
            # break

    def _execute_lammps_replacement_approximation(self, cell_to_granulate : tuple):
        """
        Replace atoms with new one.
        """
        (x_min, x_max, y_min, y_max, z_min, z_max), atom_ids  = cell_to_granulate
        # velocities_region =  self.extractor.__get_velocities__() # TODO: extract velocities
        self._lammps_execute().command(f"region kill block {x_min - 1e-3} {x_max + 1e-3} {y_min- 1e-3} {y_max+1e-3} {z_min-1e-3} {z_max+1e-3} units box")
        self._lammps_execute().command("group cell_atoms region kill")

        lenj = len(self.communicator.__get_atom_identificators__())
        self._lammps_execute().command(f"lattice fcc {self.lattice_constant_cg}")
        # velocities_of_the_cell = self.communicator.__get_velocities__()[atom_ids]
        atom_ids = self.communicator._extract_ids_from_block((x_min- 1e-3, x_max + 1e-3, y_min - 1e-3, y_max +1e-3, z_min - 1e-3, z_max+1e-3))
        velocities_of_the_cell = self.communicator.__get_velocities__()[atom_ids]
        
        mean_vx = np.mean(velocities_of_the_cell[:, 0]) * 8
        mean_vy = np.mean(velocities_of_the_cell[:, 1]) * 8
        mean_vz = np.mean(velocities_of_the_cell[:, 2]) * 8

        commands = [
            # f"variable vx_new equal {mean_vx}",
            # f"variable vy_new equal {mean_vy}",
            # f"variable vz_new equal {mean_vz}",
            'delete_atoms region kill',
            'create_atoms 2 region kill',
            # 'run 5',
            # "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
            # "variable vx_new delete",
            # "variable vy_new delete",
            # "variable vz_new delete",
        ]
        for cmd in commands:
            self._lammps_execute().command(cmd)
        self._lammps_execute().command("group cell_atoms delete")
        self._lammps_execute().command("region kill delete")
        # self._lammps_execute().command("dump_modify 1 append yes")
        self._lammps_execute().command("write_dump all custom TEST_APPROXIMATED_CRACK_dump_accurate.crack_GRAIN.lammpstrj id type x y z modify append yes")
        # self._lammps_execute().command("reset_atoms id")
        # self._lammps_execute().command("delete_atoms overlap 0.01 all all")
        # return
        if self.__DEBUG_MODE__:
            self.__debug_grained_cells.append((x_min, x_max, y_min, y_max, z_min, z_max))

    def _execute_lammps_replacement_granulation(self, cell_to_granulate : tuple):
        (x_min, x_max, y_min, y_max, z_min, z_max), atom_ids  = cell_to_granulate
        # velocities_region =  self.extractor.__get_velocities__() # TODO: extract velocities
        self._lammps_execute().command(f"region kill block {x_min - 1e-3} {x_max+1e-3} {y_min} {y_max} {z_min} {z_max} units box")
        self._lammps_execute().command("group cell_atoms region kill")

        self._lammps_execute().command(f"lattice fcc {self.lattice_constant}")
        # velocities_of_the_cell = self.communicator.__get_velocities__()[atom_ids]
        velocities_of_the_cell = self.communicator._extract_ids_from_block((x_min, x_max, y_min, y_max, z_min, z_max))
        mean_vx = 0.# np.mean(velocities_of_the_cell[:, 0]) / 8
        mean_vy = 0 #np.mean(velocities_of_the_cell[:, 1]) / 8
        mean_vz = 0 #np.mean(velocities_of_the_cell[:, 2]) / 8

        commands = [
            f"variable vx_new equal {mean_vx}",
            f"variable vy_new equal {mean_vy}",
            f"variable vz_new equal {mean_vz}",
            'delete_atoms region kill',
            'create_atoms 1 region kill',
            # 'run 5'
            "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
            "variable vx_new delete",
            "variable vy_new delete",
            "variable vz_new delete",
        ]
        for cmd in commands:
            self._lammps_execute().command(cmd)
        self._lammps_execute().command("group cell_atoms delete")
        self._lammps_execute().command("region kill delete")
        self._lammps_execute().command("write_dump all custom TEST_APPROXIMATED_CRACK_dump_accurate.crack_GRAIN.lammpstrj id type x y z modify append yes")
        # self._lammps_execute().command("delete_atoms overlap 0.01 all all")
        # return
        # if self.__DEBUG_MODE__:
        #     self.__debug_grained_cells.append((x_min, x_max, y_min, y_max, z_min, z_max))
        

    def _get_debug_info(self):
        return self.__debug_grained_cells