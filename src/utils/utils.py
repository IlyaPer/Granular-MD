from typing import Any

import lammps
import numpy as np

class LammpsCommunicator():
    def __init__(self, lammps_instance : lammps.lammps) -> None:
        self.lammps_instance = lammps_instance
        self.nlocal = lammps_instance.extract_global("nlocal")
        self.mean_positions = np.array([])

    def get_instance(self,) -> lammps.lammps:
        return self.lammps_instance
    
    def set_positions(self, mean_positions : np.ndarray):
        self.mean_positions = mean_positions

    def __get_positions__(self, current_snapshot=False) -> np.ndarray:
        raw_positions = self.lammps_instance.numpy.extract_atom("x")[:self.nlocal]
        sort_indices = np.argsort(self.__get_atom_identificators__())
        if current_snapshot:
            raw_velocities = self.lammps_instance.numpy.extract_atom("v")[:self.nlocal]
            return raw_velocities[sort_indices]
        return raw_positions[sort_indices]
    
    def get_masses(self) -> np.ndarray:
        raw_masses = self.lammps_instance.numpy.extract_atom("mass")
        types = self.__get_atom_types__()
        masses = np.array([raw_masses[t] for t in types])
        # masses = np.array([58.6934 if t==1 else 469.5472 for t in types])
        return masses
    
    def get_total_mass(self) -> np.ndarray:
        return np.sum(self.get_masses())
    
    def __get_atom_types__(self,) -> np.ndarray:
        raw_types = self.lammps_instance.numpy.extract_atom("type")[:self.nlocal]
        sort_indices = np.argsort(self.__get_atom_identificators__())
        return raw_types[sort_indices]
    
    def __get_velocities__(self,) -> np.ndarray:
        raw_velocities = self.lammps_instance.numpy.extract_atom("v")[:self.nlocal]
        sort_indices = np.argsort(self.__get_atom_identificators__())
        return raw_velocities[sort_indices]
    
    def __get_atom_identificators__(self,) -> np.ndarray:
        raw_ids = self.lammps_instance.numpy.extract_atom("id")[:self.nlocal]
        return raw_ids# LAMMPS starts indexing from 1, not 0.
    
    def __get_box_size__(self,) -> tuple:
        box = self.lammps_instance.extract_box() # TODO: MINIMAL POSITIONS INSTEAD OF BOX
        # xlo, xhi = box[0][0], box[1][0]          # TODO: WHY IS IT SHIT WITH SHRINK-WRAPPED?
        # ylo, yhi = box[0][1], box[1][1]
        # zlo, zhi = box[0][2], box[1][2]
        xlo, xhi = np.min(self.__get_positions__()[:,0]), np.max(self.__get_positions__()[:,0]) 
        ylo, yhi = np.min(self.__get_positions__()[:,1]), np.max(self.__get_positions__()[:,1])
        zlo, zhi = np.min(self.__get_positions__()[:,2]), np.max(self.__get_positions__()[:,2])
        return xlo, xhi, ylo, yhi, zlo, zhi
    
    def __get_pe_per_atom__(self,) -> tuple:
       sort_indices = np.argsort(self.__get_atom_identificators__())
       return self.lammps_instance.numpy.extract_compute('pe_atom', 1, 1)[sort_indices]
    
    def _extract_ids_from_block(self, borders : tuple) -> np.ndarray:
        x_min, x_max, y_min, y_max, z_min, z_max = borders
        positions = self.__get_positions__()
        mask = (
            (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) &
            (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max) &
            (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
        )
        identificators = self.__get_atom_identificators__()[np.where(mask)[0]]
        return identificators
    