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
        raw_masses = self.lammps_instance.numpy.extract_atom("m")[:self.nlocal]
        sort_indices = np.argsort(self.__get_atom_identificators__())
        return raw_masses[sort_indices]

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


# def extract_interesting_regions(self):
#     self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi, self.pbc = (
#         self.lammps_extractor.__get_box_size__()
#     )
#     self.clear_extractor()
#     cell_size = self.lattice_constant * 2.0  # 7.04

#     positions = self.lammps_extractor.__get_positions__()
#     atom_ids = self.lammps_extractor.__get_atom_identificators__()

#     # 1. Жесткое разбиение через floor. Граница принадлежит ТОЛЬКО одной ячейке.
#     ix = np.floor((positions[:, 0] - self.xlo) / cell_size).astype(int)
#     iy = np.floor((positions[:, 1] - self.ylo) / cell_size).astype(int)
#     iz = np.floor((positions[:, 2] - self.zlo) / cell_size).astype(int)

#     nx = int(np.round((self.xhi - self.xlo) / cell_size))
#     ny = int(np.round((self.yhi - self.ylo) / cell_size))
#     nz = int(np.round((self.zhi - self.zlo) / cell_size))

#     # 2. Исключаем граничные ячейки (как ты и хотел)
#     valid = (ix >= 1) & (ix < nx-1) & \
#             (iy >= 1) & (iy < ny-1) & \
#             (iz >= 1) & (iz < nz-1)

#     ix, iy, iz = ix[valid], iy[valid], iz[valid]
#     atom_ids = atom_ids[valid]

#     # 3. Группируем атомы по уникальным ячейкам (O(N), мгновенно)
#     cell_idx = ix * (ny * nz) + iy * nz + iz
#     unique_cells, inverse = np.unique(cell_idx, return_inverse=True)

#     for cid in unique_cells:
#         mask = inverse == cid
#         ids_in_cell = atom_ids[mask]

#         # Восстанавливаем точные границы ячейки
#         ci = int(cid // (ny * nz))
#         cj = int((cid // nz) % ny)
#         ck = int(cid % nz)

#         cell = (
#             self.xlo + ci * cell_size, self.xlo + (ci+1) * cell_size,
#             self.ylo + cj * cell_size, self.ylo + (cj+1) * cell_size,
#             self.zlo + ck * cell_size, self.zlo + (ck+1) * cell_size
#         )
        
        # Твой существующий обработчик
        self._process_single_cell(cell, ids_in_cell)
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
    