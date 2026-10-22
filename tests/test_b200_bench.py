"""CLI checks run without a GPU; opt in to device tests with RUN_CUDA_TESTS=1."""
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("BENCHMARK_BINARY", ROOT / "green_ctx_bw_bench"))


@unittest.skipUnless(BINARY.is_file(), "build the benchmark first")
class B200CliTests(unittest.TestCase):
    def test_help_exposes_validation_and_tma_tuning_without_a_gpu(self):
        result = subprocess.run([str(BINARY), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--validate-only", "--tma-stages", "--blocks-per-sm", "--sm-count"):
            self.assertIn(option, result.stdout)

    def test_invalid_parameters_are_rejected_before_cuda_initialization(self):
        cases = [
            (["0"], "buffer_MB"),
            (["-1"], "buffer_MB"),
            (["512", "0"], "iterations"),
            (["512", "1", "-1"], "gpu_id"),
            (["512", "1", "0", "0", "0"], "trials"),
            (["512", "1", "0", "0", "1", "6"], "load_mode"),
            (["--tma-stages", "3"], "tma-stages"),
            (["--blocks-per-sm", "0"], "blocks-per-sm"),
            (["--sm-count", "-1"], "sm-count"),
            (["--unknown"], "Unknown option"),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                result = subprocess.run([str(BINARY), *args], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("cuInit", result.stderr)


@unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "requires a CUDA GPU")
class B200DeviceTests(unittest.TestCase):
    def test_every_load_mode_validates_varied_input_and_partial_tiles(self):
        for mode in range(6):
            for stages in ((2, 4, 8) if mode == 4 else (2,)):
                with self.subTest(mode=mode, stages=stages):
                    result = subprocess.run(
                        [str(BINARY), "--validate-only", "--tma-stages", str(stages),
                         "1", "1", "0", "0", "1", str(mode)],
                        capture_output=True, text=True, timeout=120,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    record = json.loads(result.stdout)
                    self.assertEqual(record["kind"], "validation")
                    self.assertGreaterEqual(record["cases"], 8)
                    self.assertGreater(record["checked_float4s"], 16384)

    def test_single_green_context_reports_matching_sm_residency(self):
        result = subprocess.run(
            [str(BINARY), "--sm-count", "8", "1", "1", "0", "0", "1", "4"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["kind"], "single")
        self.assertEqual(record["sm_count_allocated"], 8)
        self.assertEqual(record["probe_observed_sm_count"], 8)
        self.assertGreater(record["bandwidth_GBps"], 0)
