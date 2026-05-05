import tempfile
import unittest

import numpy as np

from src.cg import (
    BlockAnalyzer,
    BlockTiler,
    CoarseningPolicy,
    FccCoarseGrainingFramework,
    LatticeRegistry,
    ReversibleMapper,
)
from src.cg.fcc import FCC_BASIS_Q, JsonlMetrics
from src.modifiers.changer import DynamicChanger


def fcc_positions(a=3.52, cells=(2, 2, 2), origin=(0.0, 0.0, 0.0)):
    origin = np.asarray(origin, dtype=float)
    positions = []
    for ix in range(cells[0]):
        for iy in range(cells[1]):
            for iz in range(cells[2]):
                base_q = 2 * np.array([ix, iy, iz], dtype=int)
                positions.extend(origin + 0.5 * a * (base_q + FCC_BASIS_Q))
    return np.array(positions, dtype=float)


class TestLatticeRegistry(unittest.TestCase):
    def test_registers_ideal_and_noisy_fcc_sites(self):
        rng = np.random.default_rng(42)
        positions = fcc_positions(cells=(3, 3, 3))
        noisy = positions + rng.normal(scale=0.02, size=positions.shape)

        registry = LatticeRegistry(lattice_constant=3.52, use_spglib=False).register(noisy)
        q, errors = registry.site_indices(noisy)

        self.assertTrue(np.all(registry.is_fcc_site(q)))
        self.assertLess(float(np.max(errors)), registry.tolerance)


class TestBlockTiler(unittest.TestCase):
    def test_scale_two_small_block_maps_32_to_4(self):
        positions = fcc_positions(cells=(2, 2, 2))
        registry = LatticeRegistry(lattice_constant=3.52, use_spglib=False).register(positions)
        tiler = BlockTiler(registry, block_cells=2, scale_factor=2)

        blocks = tiler.build_blocks(positions)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0].expected_q), 32)
        self.assertEqual(len(blocks[0].atom_ids), 32)
        self.assertEqual(len(blocks[0].coarse_q), 4)
        self.assertEqual(len(blocks[0].missing_q), 0)

    def test_larger_block_maps_256_to_32(self):
        positions = fcc_positions(cells=(4, 4, 4))
        registry = LatticeRegistry(lattice_constant=3.52, use_spglib=False).register(positions)
        tiler = BlockTiler(registry, block_cells=4, scale_factor=2)

        blocks = tiler.build_blocks(positions)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0].expected_q), 256)
        self.assertEqual(len(blocks[0].atom_ids), 256)
        self.assertEqual(len(blocks[0].coarse_q), 32)

    def test_missing_atoms_mark_crack_like_block(self):
        positions = fcc_positions(cells=(2, 2, 2))[3:]
        registry = LatticeRegistry(lattice_constant=3.52, use_spglib=False).register(positions)
        tiler = BlockTiler(registry, block_cells=2, scale_factor=2)
        analyzer = BlockAnalyzer()

        block = tiler.build_blocks(positions)[0]
        features = analyzer.analyze(block)

        self.assertGreater(block.missing_ratio, 0.0)
        self.assertTrue(features.has_defect)


class TestMappingAndPolicy(unittest.TestCase):
    def test_round_trip_mapping_reconstructs_fine_positions(self):
        positions = fcc_positions(cells=(2, 2, 2))
        registry = LatticeRegistry(lattice_constant=3.52, use_spglib=False).register(positions)
        block = BlockTiler(registry, block_cells=2, scale_factor=2).build_blocks(positions)[0]
        mapper = ReversibleMapper(registry, scale_factor=2)

        coarsen = mapper.coarsen_action(block)
        mapper.commit(coarsen)
        refine = mapper.refine_action(block.index, delete_atom_ids=np.arange(4))

        reconstructed = np.array(sorted(map(tuple, refine.create_positions)))
        original = np.array(sorted(map(tuple, registry.positions_from_q(block.expected_q))))

        self.assertTrue(np.allclose(reconstructed, original))

    def test_policy_uses_hysteresis_and_blocks_defects(self):
        positions = fcc_positions(cells=(2, 2, 2))
        registry = LatticeRegistry(lattice_constant=3.52, use_spglib=False).register(positions)
        block = BlockTiler(registry, block_cells=2, scale_factor=2).build_blocks(positions)[0]
        features = BlockAnalyzer().analyze(block)
        policy = CoarseningPolicy(coarsen_score=2.5, stable_checks=2)

        self.assertEqual(policy.choose(features), "keep")
        self.assertEqual(policy.choose(features), "coarsen")


class TestFramework(unittest.TestCase):
    def test_framework_dry_run_plans_coarsening_and_writes_metrics(self):
        positions = fcc_positions(cells=(2, 2, 2))
        atom_ids = np.arange(len(positions), dtype=int)
        atom_types = np.ones(len(positions), dtype=int)

        with tempfile.TemporaryDirectory() as tmp:
            framework = FccCoarseGrainingFramework(
                lattice_constant=3.52,
                scale_factor=2,
                block_cells=2,
                dry_run=True,
                metrics=JsonlMetrics(f"{tmp}/metrics.jsonl"),
            )
            actions = framework.plan(positions, atom_ids, atom_types)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "coarsen")
        self.assertEqual(len(actions[0].delete_atom_ids), 32)
        self.assertEqual(len(actions[0].create_positions), 4)
        self.assertEqual(framework.mapper.records, {})

    def test_dynamic_changer_dry_run_uses_framework_contract(self):
        positions = fcc_positions(cells=(2, 2, 2))

        class FakeCommunicator:
            def __get_positions__(self, current_snapshot=False):
                return positions

            def __get_atom_identificators__(self):
                return np.arange(len(positions), dtype=int)

            def __get_atom_types__(self):
                return np.ones(len(positions), dtype=int)

            def __get_velocities__(self):
                return np.zeros_like(positions)

            def __get_pe_per_atom__(self, required=True):
                return None

            def __get_box_size__(self):
                return (0.0, 7.04, 0.0, 7.04, 0.0, 7.04, [1, 1, 1])

            def get_instance(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            changer = DynamicChanger(
                communicator=FakeCommunicator(),
                lattice_constant=3.52,
                scale_factor=2,
                block_cells=2,
                dry_run=True,
                metrics=JsonlMetrics(f"{tmp}/metrics.jsonl"),
            )
            actions = changer.accelerate()

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "coarsen")


if __name__ == "__main__":
    unittest.main()
