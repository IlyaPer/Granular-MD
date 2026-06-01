APPROXIMATE = 1
GRANULATE = 2
import logging
from re import S
import numpy as np

from src.extractors.extractors import FccCellsExtractor
from src.utils.utils import LammpsCommunicator

TIME_WINDOW=10

class DynamicChanger():
    def __init__(self, communicator : LammpsCommunicator, extractor : FccCellsExtractor, lattice_constant : float, scale_factor : int, baby_mode=False):
        self.extractor = extractor
        self.communicator = communicator
        self.baby_mode = baby_mode
        self.lattice_constant_cg = lattice_constant * scale_factor
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

        if len(ids_to_delete) == 0:
            return
        
        #=========== Delete extra atoms from simulation ===============

        # self.extractor.get_lammps_instance().command("write_dump all custom records/TEST_images.lammpstrj id type x y z modify append yes")

        self.extractor.get_lammps_instance().command(f"group to_delete id {' '.join(list(map(str, ids_to_delete.astype(int))))}")
        self.extractor.get_lammps_instance().command(f"delete_atoms group to_delete")
        self.extractor.get_lammps_instance().command(f"group to_delete delete")



        #=========== phantom_atoms_big ===============

        for x,y,z in positions_to_spawn_atoms:
                self.extractor.get_lammps_instance().command(f"create_atoms 1 single {x} {y} {z} units box")

        
        # self.extractor.get_lammps_instance().command(f"velocity all set 10")
        # self.extractor.get_lammps_instance().command("write_dump all custom records/TEST_images.lammpstrj id type x y z modify append yes")


        #TODO velocity set
        if not only_approximate:
            for x,y,z in positions_to_spawn_atoms:
                self.extractor.get_lammps_instance().command(f"create_atoms 1 single {x} {y} {z} units box")

        for x,y,z in positions_for_large_particles:
            self.extractor.get_lammps_instance().command(f"create_atoms 2 single {x} {y} {z} units box")


        # self.extractor.get_lammps_instance().command(f"group scale_1_atoms type 1")
        # self.extractor.get_lammps_instance().command(f"group scale_2_atoms type 2")
        
        # self.extractor.get_lammps_instance().command(f"neigh_modify exclude group scale_1_atoms scale_2_atoms")

        # self.extractor.get_lammps_instance().command(f"delete_atoms overlap 0.1 all all")
        # self.extractor.get_lammps_instance().command("minimize  1.0e-8 1.0e-8 10000 100000")
        # self.extractor.get_lammps_instance().command("velocity all create 300.0 54321 rot yes")
        # self.extractor.get_lammps_instance().command("write_dump all custom records/TEST_images.lammpstrj id type x y z modify append yes")
        # self._lammps_execute().command("minimize 1e-8 1e-8 10000 100000")

        

    def my_force_callback(self, lmp_instance, ntimestep, nlocal, tag, x, fext):
        lmp = self.extractor.get_lammps_instance()
        
        f = lmp.numpy.extract_atom("f")
        atom_masks = lmp.numpy.extract_atom("mask")
        
        raw_mask = lmp.extract_global("group_extract_group")
        if raw_mask is None: return
        
        group_bitmask = int(raw_mask)
        group_indices = np.where((atom_masks & group_bitmask) != 0)[0]

        if len(group_indices) > 0:
            fext[group_indices] = (self.scale_factor - 1.0) * f[group_indices]

    def run_with_scaled_potential(self, scale_factor=2.0, iter_steps=500):
        lmp = self.extractor.get_lammps_instance()
        self.current_scale_factor = scale_factor

        lmp.set_fix_external_callback("scale_all", self.my_force_callback)

        lmp.command(f"run {iter_steps}")

    def _get_debug_info(self):
        return self.__debug_grained_cells