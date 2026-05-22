APPROXIMATE = 1
GRANULATE = 2
import logging
from re import S
import numpy as np

from src.extractors.extractors import FccCellsExtractor
from src.utils.utils import LammpsCommunicator

TIME_WINDOW=10

class DynamicChanger():
    def __init__(self, communicator : LammpsCommunicator, extractor : FccCellsExtractor, lattice_constant : float, lattice_constant_cg : float, scale_factor : int, baby_mode=False):
        self.extractor = extractor
        self.communicator = communicator
        self.baby_mode = baby_mode
        self.lattice_constant_cg = lattice_constant_cg
        self.lattice_constant = lattice_constant
        self.__debug_grained_cells = []
        self.__DEBUG_MODE__ = True
        self.scale_factor = scale_factor
        
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
        # self.tree_ids_atoms = 

    def accelerate(self, lammps_instance, only_approximate=False, only_granulate=False):
        """
        affirmative. execute acceleration.
        """
        self.iter_number = 0
        self.extractor.set_communicator(lammps_instance)

        # self.extractor.extract_symmetry(self.scale_factor)
        self.extractor.get_cells_to_apply_action()

        atoms_to_change_with_particles_1, atoms_to_change_with_particles_2 = [], []
        positions_for_large_particles = []
        positions_to_spawn_atoms = []

        if not only_granulate:
            atoms_to_change_with_particles_1, positions_for_large_particles = self.extractor.get_data_of_cells_to_approximate()

        if not only_approximate:
            atoms_to_change_with_particles_2, positions_to_spawn_atoms = self.extractor.get_data_of_cells_to_granulate()

        ids_to_delete = np.concatenate([atoms_to_change_with_particles_1, atoms_to_change_with_particles_2])
        self.extractor.get_lammps_instance().command("write_dump all custom records/TEST_images.lammpstrj id type x y z modify append yes")

        self.extractor.get_lammps_instance().command(f"group to_delete id {' '.join(list(map(str, ids_to_delete.astype(int))))}")
        self.extractor.get_lammps_instance().command(f"delete_atoms group to_delete")
        self.extractor.get_lammps_instance().command(f"group to_delete delete")

        
        # self.extractor.get_lammps_instance().command(f"velocity all set 10")
        self.extractor.get_lammps_instance().command("write_dump all custom records/TEST_images.lammpstrj id type x y z modify append yes")
        #TODO velocity set
        if not only_approximate:
            for x,y,z in positions_to_spawn_atoms:
                self.extractor.get_lammps_instance().command(f"create_atoms 1 single {x} {y} {z} units box")

        for x,y,z in positions_for_large_particles:
            self.extractor.get_lammps_instance().command(f"create_atoms 2 single {x} {y} {z} units box")

        # self.extractor.get_lammps_instance().command(f"delete_atoms overlap 0.1 all all")
        self.extractor.get_lammps_instance().command("write_dump all custom records/TEST_images.lammpstrj id type x y z modify append yes")
        # self._lammps_execute().command("minimize 1e-8 1e-8 10000 100000")

    def _execute_lammps_replacement_approximation(self, cell_to_granulate : tuple):
        """
        Replace atoms with new one.
        """
        (x_min, x_max, y_min, y_max, z_min, z_max), atom_ids  = cell_to_granulate
        # velocities_region =  self.extractor.__get_velocities__() # TODO: extract velocities
        self._lammps_execute().command(f"region kill block {x_min} {x_max} {y_min} {y_max} {z_min} {z_max} units box")
        self._lammps_execute().command("group cell_atoms region kill")

        # lenj = len(self.communicator.__get_atom_identificators__())
        self._lammps_execute().command(f"lattice fcc {self.lattice_constant_cg}")
        # velocities_of_the_cell = self.communicator.__get_velocities__()[atom_ids]
        atom_ids = self.communicator._extract_ids_from_block((x_min- 1e-3, x_max + 1e-3, y_min - 1e-3, y_max +1e-3, z_min - 1e-3, z_max+1e-3))
        # velocities_of_the_cell = self.communicator.__get_velocities__()[atom_ids]
        
        # mean_vx = np.mean(velocities_of_the_cell[:, 0]) * 8
        # mean_vy = np.mean(velocities_of_the_cell[:, 1]) * 8
        # mean_vz = np.mean(velocities_of_the_cell[:, 2]) * 8
        half_lat = self.lattice_constant_cg / 2.0

        commands = [
            # f"variable vx_new equal {mean_vx}",
            # f"variable vy_new equal {mean_vy}",
            # f"variable vz_new equal {mean_vz}",
            'delete_atoms region kill',
            f'create_atoms 2 single {x_min} {y_min} {z_min} units box',
            f'create_atoms 2 single {x_min + half_lat} {y_min + half_lat} {z_min} units box',
            f'create_atoms 2 single {x_min + half_lat} {y_min} {z_min + half_lat} units box',
            f'create_atoms 2 single {x_min} {y_min + half_lat} {z_min + half_lat} units box',
            'run 0',
            # "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
            # "variable vx_new delete",
            # "variable vy_new delete",
            # "variable vz_new delete",
        ]
        for cmd in commands:
            self._lammps_execute().command(cmd)

        box = (x_min-1e-3, x_max+1e-3, y_min-1e-3, y_max+1e-3, z_min-1e-3, z_max+1e-3)
        proc_ids = self.communicator._extract_ids_from_block(box)
        actual_count = len(proc_ids)

        # assert actual_count == 4, (
        #     f"Ожидалось 4 атомы в блоке, получено {actual_count}. "
        #     f"Box: ({x_min:.3f}, {x_max:.3f}, {y_min:.3f}, {y_max:.3f}, "
        #     f"{z_min:.3f}, {z_max:.3f}). "
        #     f"Найденные ID: {sorted(proc_ids)}"
        # )

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
        self._lammps_execute().command(f"region kill block {x_min} {x_max} {y_min} {y_max} {z_min} {z_max} units box")
        self._lammps_execute().command("group cell_atoms region kill")

        self._lammps_execute().command(f"lattice fcc {self.lattice_constant}")
        # velocities_of_the_cell = self.communicator.__get_velocities__()[atom_ids]
        velocities_of_the_cell = self.communicator._extract_ids_from_block((x_min, x_max, y_min, y_max, z_min, z_max))
        mean_vx = 0.# np.mean(velocities_of_the_cell[:, 0]) / 8
        mean_vy = 0 #np.mean(velocities_of_the_cell[:, 1]) / 8
        mean_vz = 0 #np.mean(velocities_of_the_cell[:, 2]) / 8

        commands = [
            # f"variable vx_new equal {mean_vx}",
            # f"variable vy_new equal {mean_vy}",
            # f"variable vz_new equal {mean_vz}",
            'delete_atoms region kill',
            'create_atoms 1 region kill',
            # 'run 5'
            # "velocity cell_atoms set ${vx_new} ${vy_new} ${vz_new}",
            # "variable vx_new delete",
            # "variable vy_new delete",
            # "variable vz_new delete",
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