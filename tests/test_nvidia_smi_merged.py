#!/usr/bin/env python3
import importlib.machinery
import importlib.util
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader("sidecar_smi", str(ROOT / "lib" / "nvidia-smi-merged"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
smi = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(smi)


class NvidiaSmiMergedTests(unittest.TestCase):
    def setUp(self):
        self.host = [("GPU-host", "00000000:03:00.0")]
        self.leg = [("GPU-v100", "00000000:04:00.0")]

    def test_split_id_combines_every_supported_spelling(self):
        rest, ident = smi.split_id(["-q", "-i", "0", "--id=GPU-v100", "-i=0000:04:00.0"])
        self.assertEqual(rest, ["-q"])
        self.assertEqual(ident, "0,GPU-v100,0000:04:00.0")

    def test_route_ids_maps_merged_indices_and_rejects_unknowns(self):
        self.assertEqual(smi.route_ids("0,1", self.host, self.leg), {smi.HOST: ["0"], smi.LEG: ["0"]})
        with self.assertRaisesRegex(ValueError, "outside the merged range"):
            smi.route_ids("2", self.host, self.leg)
        with self.assertRaisesRegex(ValueError, "does not match"):
            smi.route_ids("GPU-nope", self.host, self.leg)

    def test_inventory_failure_is_not_silently_an_empty_driver(self):
        with mock.patch.object(smi, "run", return_value=(9, "", "driver exploded")):
            with self.assertRaisesRegex(RuntimeError, "inventory query failed"):
                smi.gpus(smi.LEG)

    def test_snapshot_preserves_a_side_failure(self):
        results = {
            smi.HOST: (0, "GPU 0: host\n", ""),
            smi.LEG: (7, "", "legacy failed\n"),
        }
        with mock.patch.object(smi, "run", side_effect=lambda side, args: results[side]):
            rc, out, err = smi.snapshot(["-L"], 1)
        self.assertEqual(rc, 7)
        self.assertIn("GPU 0: host", out)
        self.assertIn("legacy failed", err)


if __name__ == "__main__":
    unittest.main()
