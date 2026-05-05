import logging

import numpy as np

from src.cg import FccCoarseGrainingFramework
from src.utils.utils import LammpsCommunicator

logger = logging.getLogger(__name__)


class DynamicChanger:
    def __init__(
        self,
        communicator: LammpsCommunicator,
        lattice_constant: float = 3.52,
        scale_factor: int = 2,
        block_cells: int = 2,
        dry_run: bool = True,
        metrics=None,
    ):
        self.communicator = communicator
        self.lattice_constant = lattice_constant
        self.scale_factor = scale_factor
        self.block_cells = block_cells
        self.dry_run = dry_run
        self.framework = FccCoarseGrainingFramework(
            lattice_constant=lattice_constant,
            scale_factor=scale_factor,
            block_cells=block_cells,
            dry_run=dry_run,
            metrics=metrics,
        )
        self.__debug_actions = []

    def set_lammps(self, lammps_instance):
        self.communicator = LammpsCommunicator(lammps_instance)

    def _lammps_execute(self):
        return self.communicator.get_instance()

    def accelerate(self, lammps_instance=None):
        if lammps_instance is not None:
            self.set_lammps(lammps_instance)

        positions = self.communicator.__get_positions__(current_snapshot=True)
        atom_ids = self.communicator.__get_atom_identificators__()
        atom_types = self.communicator.__get_atom_types__()
        velocities = self.communicator.__get_velocities__()
        pe_atom = self.communicator.__get_pe_per_atom__(required=False)
        box = self.communicator.__get_box_size__()

        actions = self.framework.plan(
            positions=positions,
            atom_ids=atom_ids,
            atom_types=atom_types,
            velocities=velocities,
            pe_atom=pe_atom,
            box=box,
        )
        self.__debug_actions.extend(actions)

        if self.dry_run:
            logger.info("CG dry-run planned %s actions.", len(actions))
            return actions

        for action in actions:
            self._apply_action(action)
        self.framework.commit_actions(actions)
        return actions

    def _apply_action(self, action):
        if len(action.delete_atom_ids):
            ids = " ".join(map(str, np.asarray(action.delete_atom_ids, dtype=int) + 1))
            self._lammps_execute().command(f"group cg_delete id {ids}")
            self._lammps_execute().command("delete_atoms group cg_delete")
            self._lammps_execute().command("group cg_delete delete")

        for position in action.create_positions:
            x, y, z = map(float, position)
            self._lammps_execute().command(
                f"create_atoms {action.create_type} single {x} {y} {z} units box"
            )

        self._lammps_execute().command("reset_atoms id")

    def _get_debug_info(self):
        return self.__debug_actions