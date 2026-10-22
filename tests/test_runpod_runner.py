import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_b200_runpod.py"


class NsightReportTests(unittest.TestCase):
    def parse_report(self, text):
        spec = importlib.util.spec_from_file_location("runpod_report", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "parse_ncu_report"), "profiler artifacts must be validated")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.csv"
            path.write_text(text)
            return module.parse_ncu_report(path, 2097152)

    def test_report_converts_units_and_compares_actual_dram_bytes(self):
        result = self.parse_report(
            '==PROF== Profiling kernel\n'
            '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"0","read_bandwidth_kernel_tma<4>","dram__bytes_read.sum","Mbyte","2.097152"\n'
            '"0","read_bandwidth_kernel_tma<4>","dram__bytes_write.sum","byte","1024"\n'
            '"0","read_bandwidth_kernel_tma<4>","gpu__time_duration.sum","usecond","100"\n'
            '"0","read_bandwidth_kernel_tma<4>","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","92"\n'
        )
        self.assertEqual(result["dram_read_bytes"], 2097152)
        self.assertEqual(result["dram_read_fraction_of_input"], 1.0)
        self.assertAlmostEqual(result["dram_read_GBps"], 20.97152)

    def test_missing_or_nonfinite_counters_do_not_count_as_success(self):
        with self.assertRaisesRegex(RuntimeError, "counter"):
            self.parse_report("==WARNING== No kernels were profiled\n")
        with self.assertRaisesRegex(RuntimeError, "counter"):
            self.parse_report(
                '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                '"0","read_bandwidth_kernel_tma<4>","dram__bytes_read.sum","byte","nan"\n'
            )


class RunpodPreflightTests(unittest.TestCase):
    def run_preflight(self, cuda="12.8", gpu="NVIDIA B200", cc="10.0"):
        # Only external discovery commands are replaced; the runner itself executes.
        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp)
            commands = {
                "nvcc": f"#!/bin/sh\necho 'Cuda compilation tools, release {cuda}, V{cuda}.90'\n",
                "nvidia-smi": f"#!/bin/sh\necho '0, {gpu}, GPU-test-b200, {cc}, 570.124.06, 183360'\n",
            }
            for name, text in commands.items():
                path = bindir / name
                path.write_text(text)
                path.chmod(0o755)
            environment = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
            return subprocess.run(
                ["python3", str(RUNNER), "--preflight-only"],
                capture_output=True, text=True, env=environment,
            )

    def test_b200_with_cuda_12_8_is_accepted_and_identified_by_uuid(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["gpu"]["uuid"], "GPU-test-b200")
        self.assertEqual(data["cuda_arch"], "sm_100")

    def test_old_toolkit_is_rejected_before_building(self):
        result = self.run_preflight(cuda="12.4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA 12.8", result.stderr)

    def test_wrong_gpu_is_rejected_before_building(self):
        result = self.run_preflight(gpu="NVIDIA H100", cc="9.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("B200", result.stderr)

    def test_newer_toolkit_is_accepted(self):
        result = self.run_preflight(cuda="13.0")
        self.assertEqual(result.returncode, 0, result.stderr)
