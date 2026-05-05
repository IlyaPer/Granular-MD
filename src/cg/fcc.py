from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

FCC_BASIS_Q = np.array(
    [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ],
    dtype=int,
)


@dataclass
class LatticeRegistry:
    lattice_constant: float
    origin: np.ndarray | None = None
    tolerance: float = 0.25
    use_spglib: bool = True
    spacegroup: str | None = None
    registration_error: float = 0.0

    def register(self, positions: np.ndarray, box: tuple | None = None) -> "LatticeRegistry":
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            self.origin = np.zeros(3)
            return self

        if self.origin is None:
            self.origin = self._estimate_origin(positions)

        self.spacegroup = self._try_spglib(positions, box)
        _, errors = self.site_indices(positions)
        self.registration_error = float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0
        return self

    def _estimate_origin(self, positions: np.ndarray) -> np.ndarray:
        lo = np.min(positions, axis=0)
        return np.floor(lo / self.lattice_constant) * self.lattice_constant

    def _try_spglib(self, positions: np.ndarray, box: tuple | None) -> str | None:
        if not self.use_spglib:
            return None
        try:
            import spglib  # type: ignore
        except Exception:
            logger.info("spglib is not available; using manual FCC registry.")
            return None

        try:
            if box is None:
                lo = np.min(positions, axis=0)
                hi = np.max(positions, axis=0) + self.lattice_constant
            else:
                lo = np.array([box[0], box[2], box[4]], dtype=float)
                hi = np.array([box[1], box[3], box[5]], dtype=float)
            lengths = np.maximum(hi - lo, self.lattice_constant)
            lattice = np.diag(lengths)
            scaled = (positions - lo) / lengths
            scaled = scaled - np.floor(scaled)
            numbers = np.ones(len(positions), dtype=int)
            dataset = spglib.get_symmetry_dataset(
                (lattice, scaled, numbers), symprec=self.tolerance
            )
            if dataset is None:
                return None
            return str(dataset.get("international", ""))
        except Exception as exc:
            logger.debug("spglib registration failed: %s", exc)
            return None

    def site_indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.origin is None:
            raise ValueError("LatticeRegistry.register() must be called first.")
        frac2 = 2.0 * (np.asarray(positions, dtype=float) - self.origin) / self.lattice_constant
        q = np.rint(frac2).astype(int)
        reconstructed = self.positions_from_q(q)
        errors = np.linalg.norm(np.asarray(positions) - reconstructed, axis=1)
        return q, errors

    def positions_from_q(self, q: np.ndarray) -> np.ndarray:
        if self.origin is None:
            raise ValueError("LatticeRegistry.register() must be called first.")
        return self.origin + 0.5 * self.lattice_constant * np.asarray(q, dtype=float)

    @staticmethod
    def is_fcc_site(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=int)
        return np.sum(q, axis=-1) % 2 == 0


@dataclass(frozen=True)
class LatticeBlock:
    index: tuple[int, int, int]
    q_min: np.ndarray
    expected_q: np.ndarray
    coarse_q: np.ndarray
    atom_indices: np.ndarray
    atom_ids: np.ndarray
    missing_q: np.ndarray
    duplicate_atom_ids: np.ndarray
    rms_error: float

    @property
    def missing_ratio(self) -> float:
        if len(self.expected_q) == 0:
            return 1.0
        return len(self.missing_q) / len(self.expected_q)


class BlockTiler:
    def __init__(self, registry: LatticeRegistry, block_cells: int = 2, scale_factor: int = 2):
        if block_cells < scale_factor or block_cells % scale_factor != 0:
            raise ValueError("block_cells must be a multiple of scale_factor.")
        self.registry = registry
        self.block_cells = int(block_cells)
        self.scale_factor = int(scale_factor)
        self.q_span = 2 * self.block_cells

    def build_blocks(
        self,
        positions: np.ndarray,
        atom_ids: np.ndarray | None = None,
        atom_types: np.ndarray | None = None,
        atomistic_type: int = 1,
    ) -> list[LatticeBlock]:
        positions = np.asarray(positions, dtype=float)
        if atom_ids is None:
            atom_ids = np.arange(len(positions), dtype=int)
        if atom_types is None:
            atom_types = np.full(len(positions), atomistic_type, dtype=int)

        atom_mask = atom_types == atomistic_type
        source_indices = np.where(atom_mask)[0]
        if len(source_indices) == 0:
            return []

        q_all, errors_all = self.registry.site_indices(positions[source_indices])
        valid = (errors_all <= self.registry.tolerance) & self.registry.is_fcc_site(q_all)
        source_indices = source_indices[valid]
        q_all = q_all[valid]
        errors_all = errors_all[valid]

        by_block: dict[tuple[int, int, int], list[tuple[int, np.ndarray, float]]] = {}
        for src_idx, q, err in zip(source_indices, q_all, errors_all):
            cell = np.floor_divide(q, 2)
            block_idx = tuple(np.floor_divide(cell, self.block_cells).astype(int).tolist())
            by_block.setdefault(block_idx, []).append((int(src_idx), q, float(err)))

        blocks: list[LatticeBlock] = []
        for block_idx, rows in by_block.items():
            q_min = 2 * self.block_cells * np.array(block_idx, dtype=int)
            expected_q = self.expected_fine_q(q_min)
            coarse_q = self.expected_coarse_q(q_min)

            first_by_q: dict[tuple[int, int, int], int] = {}
            duplicates: list[int] = []
            errors: list[float] = []
            for src_idx, q, err in rows:
                key = tuple(q.tolist())
                if key in first_by_q:
                    duplicates.append(int(atom_ids[src_idx]))
                    continue
                first_by_q[key] = src_idx
                errors.append(err)

            expected_keys = {tuple(q.tolist()) for q in expected_q}
            present_keys = set(first_by_q)
            missing = np.array([q for q in expected_q if tuple(q.tolist()) not in present_keys], dtype=int)
            atom_indices = np.array(
                [first_by_q[key] for key in sorted(present_keys & expected_keys)], dtype=int
            )
            blocks.append(
                LatticeBlock(
                    index=block_idx,
                    q_min=q_min,
                    expected_q=expected_q,
                    coarse_q=coarse_q,
                    atom_indices=atom_indices,
                    atom_ids=np.asarray(atom_ids)[atom_indices].astype(int),
                    missing_q=missing.reshape((-1, 3)) if missing.size else np.empty((0, 3), dtype=int),
                    duplicate_atom_ids=np.array(duplicates, dtype=int),
                    rms_error=float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0,
                )
            )
        return blocks

    def expected_fine_q(self, q_min: np.ndarray) -> np.ndarray:
        q_values = []
        for ix in range(self.block_cells):
            for iy in range(self.block_cells):
                for iz in range(self.block_cells):
                    base = q_min + 2 * np.array([ix, iy, iz], dtype=int)
                    q_values.extend(base + FCC_BASIS_Q)
        return np.array(q_values, dtype=int)

    def expected_coarse_q(self, q_min: np.ndarray) -> np.ndarray:
        q_values = []
        coarse_cells = self.block_cells // self.scale_factor
        for ix in range(coarse_cells):
            for iy in range(coarse_cells):
                for iz in range(coarse_cells):
                    base = q_min + 2 * self.scale_factor * np.array([ix, iy, iz], dtype=int)
                    q_values.extend(base + self.scale_factor * FCC_BASIS_Q)
        return np.array(q_values, dtype=int)


@dataclass
class BlockFeatures:
    block_index: tuple[int, int, int]
    atom_count: int
    expected_count: int
    missing_ratio: float
    duplicate_count: int
    mean_pe: float | None
    kinetic_temperature: float
    rms_error: float
    score: float
    has_defect: bool


class BlockAnalyzer:
    def __init__(self, mass: float = 58.69, temperature_weight: float = 1e-3):
        self.mass = mass
        self.temperature_weight = temperature_weight

    def analyze(
        self,
        block: LatticeBlock,
        velocities: np.ndarray | None = None,
        pe_atom: np.ndarray | None = None,
    ) -> BlockFeatures:
        mean_pe = None
        energy_term = 0.0
        if pe_atom is not None and len(block.atom_indices):
            values = np.asarray(pe_atom)[block.atom_indices]
            mean_pe = float(np.mean(values))
            energy_term = abs(mean_pe)

        kinetic_temperature = 0.0
        if velocities is not None and len(block.atom_indices):
            v = np.asarray(velocities)[block.atom_indices]
            v_rel = v - np.mean(v, axis=0)
            kinetic_temperature = float(np.mean(np.sum(v_rel * v_rel, axis=1)))

        has_defect = (
            block.missing_ratio > 0.0
            or len(block.duplicate_atom_ids) > 0
            or block.rms_error > 0.35
        )
        score = (
            block.missing_ratio * 10.0
            + len(block.duplicate_atom_ids)
            + block.rms_error
            + energy_term
            + self.temperature_weight * kinetic_temperature
        )
        return BlockFeatures(
            block_index=block.index,
            atom_count=len(block.atom_ids),
            expected_count=len(block.expected_q),
            missing_ratio=block.missing_ratio,
            duplicate_count=len(block.duplicate_atom_ids),
            mean_pe=mean_pe,
            kinetic_temperature=kinetic_temperature,
            rms_error=block.rms_error,
            score=float(score),
            has_defect=bool(has_defect),
        )


@dataclass
class BlockAction:
    kind: str
    block_index: tuple[int, int, int]
    delete_atom_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    create_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    create_type: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    features: BlockFeatures | None = None
    fine_q: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    coarse_q: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))


class CoarseningPolicy:
    def __init__(
        self,
        coarsen_score: float = 2.5,
        refine_score: float = 6.0,
        stable_checks: int = 1,
        cooldown_steps: int = 1,
    ):
        self.coarsen_score = coarsen_score
        self.refine_score = refine_score
        self.stable_checks = stable_checks
        self.cooldown_steps = cooldown_steps
        self._stable: dict[tuple[int, int, int], int] = {}
        self._cooldown: dict[tuple[int, int, int], int] = {}

    def choose(self, features: BlockFeatures, current_level: int = 0) -> str:
        idx = features.block_index
        if self._cooldown.get(idx, 0) > 0:
            self._cooldown[idx] -= 1
            return "keep"

        if features.has_defect or features.score >= self.refine_score:
            self._stable[idx] = 0
            return "refine" if current_level > 0 else "keep"

        if current_level == 0 and features.score <= self.coarsen_score:
            self._stable[idx] = self._stable.get(idx, 0) + 1
            if self._stable[idx] >= self.stable_checks:
                self._cooldown[idx] = self.cooldown_steps
                self._stable[idx] = 0
                return "coarsen"
            return "keep"

        self._stable[idx] = 0
        return "keep"


@dataclass
class MappingRecord:
    block_index: tuple[int, int, int]
    fine_q: np.ndarray
    coarse_q: np.ndarray
    velocity: np.ndarray
    level: int = 1


class ReversibleMapper:
    def __init__(self, registry: LatticeRegistry, scale_factor: int = 2):
        self.registry = registry
        self.scale_factor = int(scale_factor)
        self.records: dict[tuple[int, int, int], MappingRecord] = {}

    def coarsen_action(
        self,
        block: LatticeBlock,
        velocities: np.ndarray | None = None,
        features: BlockFeatures | None = None,
    ) -> BlockAction:
        velocity = np.zeros(3)
        if velocities is not None and len(block.atom_indices):
            velocity = np.mean(np.asarray(velocities)[block.atom_indices], axis=0)
        return BlockAction(
            kind="coarsen",
            block_index=block.index,
            delete_atom_ids=block.atom_ids.copy(),
            create_positions=self.registry.positions_from_q(block.coarse_q),
            create_type=2,
            velocity=velocity,
            features=features,
            fine_q=block.expected_q.copy(),
            coarse_q=block.coarse_q.copy(),
        )

    def refine_action(
        self,
        block_index: tuple[int, int, int],
        delete_atom_ids: np.ndarray | None = None,
        features: BlockFeatures | None = None,
    ) -> BlockAction:
        record = self.records[block_index]
        return BlockAction(
            kind="refine",
            block_index=block_index,
            delete_atom_ids=np.asarray(delete_atom_ids if delete_atom_ids is not None else [], dtype=int),
            create_positions=self.registry.positions_from_q(record.fine_q),
            create_type=1,
            velocity=record.velocity.copy(),
            features=features,
            fine_q=record.fine_q.copy(),
            coarse_q=record.coarse_q.copy(),
        )

    def commit(self, action: BlockAction) -> None:
        if action.kind == "coarsen":
            self.records[action.block_index] = MappingRecord(
                block_index=action.block_index,
                fine_q=action.fine_q.copy(),
                coarse_q=action.coarse_q.copy(),
                velocity=action.velocity.copy(),
            )
        elif action.kind == "refine":
            self.records.pop(action.block_index, None)


class JsonlMetrics:
    def __init__(self, path: str | Path = "logs/cg_metrics.jsonl", enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled

    def emit(self, row: dict) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


class FccCoarseGrainingFramework:
    def __init__(
        self,
        lattice_constant: float,
        scale_factor: int = 2,
        block_cells: int = 2,
        tolerance: float = 0.25,
        dry_run: bool = True,
        metrics: JsonlMetrics | None = None,
    ):
        self.registry = LatticeRegistry(lattice_constant=lattice_constant, tolerance=tolerance)
        self.tiler = BlockTiler(self.registry, block_cells=block_cells, scale_factor=scale_factor)
        self.analyzer = BlockAnalyzer()
        self.policy = CoarseningPolicy()
        self.mapper = ReversibleMapper(self.registry, scale_factor=scale_factor)
        self.dry_run = dry_run
        self.metrics = metrics or JsonlMetrics(enabled=True)
        self.iteration = 0

    def plan(
        self,
        positions: np.ndarray,
        atom_ids: np.ndarray,
        atom_types: np.ndarray,
        velocities: np.ndarray | None = None,
        pe_atom: np.ndarray | None = None,
        box: tuple | None = None,
    ) -> list[BlockAction]:
        started = time.perf_counter()
        self.registry.register(positions[atom_types == 1] if np.any(atom_types == 1) else positions, box=box)
        blocks = self.tiler.build_blocks(positions, atom_ids=atom_ids, atom_types=atom_types)
        actions: list[BlockAction] = []
        scores = []
        defects = 0
        action_blocks: set[tuple[int, int, int]] = set()

        for block in blocks:
            features = self.analyzer.analyze(block, velocities=velocities, pe_atom=pe_atom)
            scores.append(features.score)
            defects += int(features.has_defect)
            current_level = 1 if block.index in self.mapper.records else 0
            decision = self.policy.choose(features, current_level=current_level)
            if decision == "coarsen":
                actions.append(self.mapper.coarsen_action(block, velocities=velocities, features=features))
                action_blocks.add(block.index)
            elif decision == "refine" and block.index in self.mapper.records:
                actions.append(self.mapper.refine_action(block.index, features=features))
                action_blocks.add(block.index)

        actions.extend(
            self._plan_refinements_for_coarse_blocks(
                positions=positions,
                atom_ids=atom_ids,
                atom_types=atom_types,
                velocities=velocities,
                pe_atom=pe_atom,
                skip_blocks=action_blocks,
            )
        )

        self.metrics.emit(
            {
                "iteration": self.iteration,
                "blocks": len(blocks),
                "actions": {"coarsen": _count(actions, "coarsen"), "refine": _count(actions, "refine")},
                "defect_blocks": defects,
                "mean_score": float(np.mean(scores)) if scores else math.nan,
                "max_score": float(np.max(scores)) if scores else math.nan,
                "registration_rms": self.registry.registration_error,
                "spacegroup": self.registry.spacegroup,
                "elapsed_sec": time.perf_counter() - started,
                "dry_run": self.dry_run,
            }
        )
        self.iteration += 1
        return actions

    def commit_actions(self, actions: Iterable[BlockAction]) -> None:
        for action in actions:
            self.mapper.commit(action)

    def _plan_refinements_for_coarse_blocks(
        self,
        positions: np.ndarray,
        atom_ids: np.ndarray,
        atom_types: np.ndarray,
        velocities: np.ndarray | None,
        pe_atom: np.ndarray | None,
        skip_blocks: set[tuple[int, int, int]],
    ) -> list[BlockAction]:
        if not self.mapper.records or not np.any(atom_types == 2):
            return []

        coarse_indices = np.where(atom_types == 2)[0]
        q_coarse, errors = self.registry.site_indices(positions[coarse_indices])
        by_q = {
            tuple(q.tolist()): (int(coarse_indices[i]), float(errors[i]))
            for i, q in enumerate(q_coarse)
            if errors[i] <= self.registry.tolerance * self.tiler.scale_factor
        }

        actions: list[BlockAction] = []
        for block_index, record in list(self.mapper.records.items()):
            if block_index in skip_blocks:
                continue
            rows = [by_q.get(tuple(q.tolist())) for q in record.coarse_q]
            present = [row for row in rows if row is not None]
            delete_ids = np.array([atom_ids[idx] for idx, _ in present], dtype=int)
            missing_ratio = 1.0 - len(present) / max(len(record.coarse_q), 1)
            mean_pe = None
            energy_term = 0.0
            present_indices = np.array([idx for idx, _ in present], dtype=int)
            if pe_atom is not None and len(present_indices):
                values = np.asarray(pe_atom)[present_indices]
                mean_pe = float(np.mean(values))
                energy_term = abs(mean_pe)
            rms_error = float(np.sqrt(np.mean([err * err for _, err in present]))) if present else 0.0
            features = BlockFeatures(
                block_index=block_index,
                atom_count=len(present),
                expected_count=len(record.coarse_q),
                missing_ratio=missing_ratio,
                duplicate_count=0,
                mean_pe=mean_pe,
                kinetic_temperature=0.0,
                rms_error=rms_error,
                score=10.0 * missing_ratio + energy_term + rms_error,
                has_defect=missing_ratio > 0.0,
            )
            decision = self.policy.choose(features, current_level=1)
            if decision == "refine":
                actions.append(
                    self.mapper.refine_action(
                        block_index, delete_atom_ids=delete_ids, features=features
                    )
                )
        return actions


def _count(actions: Iterable[BlockAction], kind: str) -> int:
    return sum(1 for action in actions if action.kind == kind)
