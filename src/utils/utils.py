import numpy as np


class LammpsCommunicator:
    def __init__(self, lammps_instance) -> None:
        self.lammps_instance = lammps_instance
        self.nlocal = lammps_instance.extract_global("nlocal")
        self.mean_positions = np.array([])

    def get_instance(self):
        return self.lammps_instance

    def _refresh_nlocal(self) -> int:
        self.nlocal = self.lammps_instance.extract_global("nlocal")
        return self.nlocal
    
    def __get_positions__(self, current_snapshot=False) -> np.ndarray:
        if current_snapshot or len(self.mean_positions) == 0:
            nlocal = self._refresh_nlocal()
            raw_positions = self.lammps_instance.numpy.extract_atom("x")[:nlocal]
            sort_indices = np.argsort(self.__get_atom_identificators__())
            return raw_positions[sort_indices]
        return self.mean_positions
    
    def __get_atom_types__(self) -> np.ndarray:
        nlocal = self._refresh_nlocal()
        raw_types = self.lammps_instance.numpy.extract_atom("type")[:nlocal]
        sort_indices = np.argsort(self.__get_atom_identificators__())
        return raw_types[sort_indices]
    
    def __get_velocities__(self) -> np.ndarray:
        nlocal = self._refresh_nlocal()
        raw_velocities = self.lammps_instance.numpy.extract_atom("v")[:nlocal]
        sort_indices = np.argsort(self.__get_atom_identificators__())
        return raw_velocities[sort_indices]
    
    def __get_atom_identificators__(self) -> np.ndarray:
        nlocal = self._refresh_nlocal()
        raw_ids = self.lammps_instance.numpy.extract_atom("id")[:nlocal]
        return raw_ids - 1
    
    def __get_box_size__(self) -> tuple:
        box = self.lammps_instance.extract_box()
        positions = self.__get_positions__()
        xlo, xhi = np.min(positions[:, 0]), np.max(positions[:, 0])
        ylo, yhi = np.min(positions[:, 1]), np.max(positions[:, 1])
        zlo, zhi = np.min(positions[:, 2]), np.max(positions[:, 2])
        return xlo, xhi, ylo, yhi, zlo, zhi, box[5]
    
    def __get_pe_per_atom__(self, required=True) -> tuple | None:
        try:
            sort_indices = np.argsort(self.__get_atom_identificators__())
            values = self.lammps_instance.numpy.extract_compute("pe_atom", 1, 1)
            if values is None:
                if required:
                    raise RuntimeError("LAMMPS compute 'pe_atom' is not defined.")
                return None
            return values[sort_indices]
        except Exception:
            if required:
                raise
            return None