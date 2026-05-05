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

        # positions = np.array(self.lammps_instance.gather_atoms("x", 1, 3))  # sorted and wrapped already?

        if current_snapshot or len(self.mean_positions) == 0:
            raw_velocities = self.lammps_instance.numpy.extract_atom("x")[:self.nlocal]
            sort_indices = np.argsort(self.__get_atom_identificators__())
            return raw_velocities[sort_indices]
        # raw_pos = np.array(self.lammps_instance.gather_atoms("x", 1, 3)).reshape(126, 3)
        # image_flags = self.lammps_instance.extract_atom("image", 2)
        # boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lammps_instance.extract_box()
        # raw_pos = self.lammps_instance.numpy.extract_atom("x")[:self.nlocal]
        # box_length = np.array([boxhi[0] - boxlo[0], boxhi[1] - boxlo[1], boxhi[2] - boxlo[2]])

        # # Convert to NumPy arrays for easy vector manipulation
        # x_unwrapped = np.ctypeslib.as_array(raw_pos, shape=(126, 3))
        # images = np.ctypeslib.as_array(image_flags, shape=(126, 3))

        # # 3. Calculate wrapped coordinates: x_wrapped = x_unwrapped - (image_flag * box_length)
        # x_wrapped = x_unwrapped - images * box_length
        # sort_indices = np.argsort(self.__get_atom_identificators__())
        return self.mean_positions

    # def __get_positions__(self) -> np.ndarray:
    #     image_flags = self.lammps_instance.extract_atom("image", 2)
    #     boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lammps_instance.extract_box()
    #     box_length = np.array([
    #         boxhi[0] - boxlo[0],
    #         boxhi[1] - boxlo[1],
    #         boxhi[2] - boxlo[2]
    #     ])

    #     # raw_pos — это уже wrapped‑координаты
    #     raw_pos = self.lammps_instance.numpy.extract_atom("x")[:self.nlocal]
    #     x_wrapped = np.ctypeslib.as_array(raw_repo, shape=(self.nlocal, 3)).copy()

    #     # если нужны именно координаты строго в box, можно сделать норму
    #     # для каждой размерности отдельно
    #     x_normalized = x_wrapped.copy()
    #     for d in range(3):
    #         L = box_length[d]
    #         # центрируем относительно boxlo
    #         x_normalized[:, d] -= boxlo[d]
    #         # нормализуем по PBC
    #         x_normalized[:, d] %= L
    #         # сдвигаем обратно в [boxlo, boxhi)
    #         x_normalized[:, d] += boxlo[d]

    #     # сортировка по идентификаторам
    #     sort_indices = np.argsort(self.__get_atom_identificators__())
    #     return x_normalized[sort_indices]
    
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
        return raw_ids - 1 # LAMMPS starts indexing from 1, not 0.
    
    def __get_box_size__(self,) -> tuple:
        box = self.lammps_instance.extract_box() # TODO: MINIMAL POSITIONS INSTEAD OF BOX
        # xlo, xhi = box[0][0], box[1][0]          # TODO: WHY IS IT SHIT WITH SHRINK-WRAPPED?
        # ylo, yhi = box[0][1], box[1][1]
        # zlo, zhi = box[0][2], box[1][2]
        xlo, xhi = np.min(self.__get_positions__()[:,0]), np.max(self.__get_positions__()[:,0]) 
        ylo, yhi = np.min(self.__get_positions__()[:,1]), np.max(self.__get_positions__()[:,1])
        zlo, zhi = np.min(self.__get_positions__()[:,2]), np.max(self.__get_positions__()[:,2])
        return xlo, xhi, ylo, yhi, zlo, zhi, box[5]
    
    def __get_pe_per_atom__(self,) -> tuple:
       sort_indices = np.argsort(self.__get_atom_identificators__())
       return self.lammps_instance.numpy.extract_compute('pe_atom', 1, 1)[sort_indices]
    
    def _extract_ids_from_block(self, borders : tuple) -> np.ndarray:
        x_min, x_max, y_min, y_max, z_min, z_max = borders
        positions = self.__get_positions__()
        mask = (
            (positions[:, 0] >= x_min - 0.1) & (positions[:, 0] <= x_max + 0.1) &
            (positions[:, 1] >= y_min- 0.1) & (positions[:, 1] <= y_max+ 0.1) &
            (positions[:, 2] >= z_min- 0.1) & (positions[:, 2] <= z_max+ 0.1)
        )
        identificators = self.__get_atom_identificators__()[np.where(mask)[0]]
        return identificators
    