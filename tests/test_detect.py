#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sidecar_detect", ROOT / "lib" / "detect.py")
detect = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(detect)


class DetectTests(unittest.TestCase):
    def test_version_matrix(self):
        self.assertEqual(detect.version_status("580.178.04"), "tested")
        self.assertEqual(detect.version_record("580.178.04")["physically_tested_ids"], ["1DB6"])
        self.assertEqual(detect.version_status("580.196.07"), "untested")
        self.assertEqual(detect.version_status("590.1"), "unsupported")
        self.assertEqual(detect.version_status("../../tmp"), "unsupported")
        self.assertEqual(
            set(detect.supported_legacy_ids()),
            {"1DB1", "1DB3", "1DB4", "1DB5", "1DB6", "1DB7", "1DB8", "1DF6"},
        )

    def test_every_v100_id_is_automatic_for_tested_and_new_r580_points(self):
        expected_ids = ("1DB1", "1DB3", "1DB4", "1DB5", "1DB6", "1DB7", "1DB8", "1DF6")
        expected = "\n".join(expected_ids) + "\n"
        for version in ("580.178.04", "580.196.07"):
            with self.subTest(version=version):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(detect.main(["release-ids", version]), 0)
                self.assertEqual(output.getvalue(), expected)
        self.assertEqual(detect.main(["release-ids", "590.1"]), 1)

    def test_parse_ids(self):
        self.assertEqual(detect.parse_ids("1db6,10de:1B80,1db6"), [0x1B80, 0x1DB6])
        self.assertEqual(detect.normalized_ids("1db6,,10DE:1DB6, 1b80"), "1B80,1DB6")
        for bad in ("", "xyz", "10000", "-1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                detect.parse_ids(bad)

    def test_count_present_ids_counts_physical_matches(self):
        original = detect.gpus
        detect.gpus = lambda: [
            {"id": "1db6", "bdf": "0000:04:00.0"},
            {"id": "1DB6", "bdf": "0000:05:00.0"},
            {"id": "2d04", "bdf": "0000:03:00.0"},
        ]
        try:
            self.assertEqual(detect.count_present_ids("1DB6"), 2)
            self.assertEqual(detect.count_present_ids("1DB6,2D04"), 3)
        finally:
            detect.gpus = original

    def test_catalog_rejects_modern_and_pre_r580_gpus(self):
        v100_ids = ("1DB1", "1DB3", "1DB4", "1DB5", "1DB6", "1DB7", "1DB8", "1DF6")
        catalog = {
            "chips": [
                *({"devid": f"0x{device_id}", "name": f"Tesla V100 family {device_id}", "features": []}
                  for device_id in v100_ids),
                {"devid": "0x2D04", "name": "RTX 5060 Ti", "features": ["kernelopen"]},
                {"devid": "0x1180", "name": "Kepler", "legacybranch": "470.xx", "features": []},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "supported-gpus.json"
            path.write_text(json.dumps(catalog))
            output = io.StringIO()
            errors = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                self.assertEqual(detect.validate_ids(path, ",".join(v100_ids)), 0)
                self.assertEqual(detect.validate_ids(path, "2d04,1180"), 1)
            self.assertIn("Tesla V100", output.getvalue())
            self.assertIn("modern GPU", errors.getvalue())
            self.assertIn("470.xx", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
