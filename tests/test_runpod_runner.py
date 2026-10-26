import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import importlib.util
from unittest.mock import patch

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
    def run_preflight(self, cuda="12.8", gpu="NVIDIA B200", cc="10.0", runner=RUNNER):
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
                ["python3", str(runner), "--preflight-only"],
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

    def test_gb200_keeps_existing_blackwell_support(self):
        result = self.run_preflight(gpu="NVIDIA GB200")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["gpu_family"], "B200")
        self.assertEqual(data["cuda_arch"], "sm_100")

    def test_h200_with_cuda_12_4_selects_native_hopper(self):
        result = self.run_preflight(cuda="12.4", gpu="NVIDIA H200", cc="9.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["gpu_family"], "H200")
        self.assertEqual(data["cuda_arch"], "sm_90")

    def test_h200_old_toolkit_is_rejected_before_building(self):
        result = self.run_preflight(cuda="12.3", gpu="NVIDIA H200", cc="9.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("H200 requires", result.stderr)
        self.assertIn("CUDA 12.4", result.stderr)

    def test_h200_with_wrong_compute_capability_is_rejected(self):
        result = self.run_preflight(gpu="NVIDIA H200", cc="10.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compute capability", result.stderr)

    def test_h200_entrypoint_uses_same_hardware_detection(self):
        result = self.run_preflight(cuda="12.8", gpu="NVIDIA H200 NVL", cc="9.0",
                                    runner=ROOT / "run_h200_runpod.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["cuda_arch"], "sm_90")


class RunpodBuildTests(unittest.TestCase):
    def test_copy_only_reaches_tma_validation_and_timing_but_not_other_modes(self):
        spec = importlib.util.spec_from_file_location("runpod_copy_only", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata = {"gpu": {"uuid": "GPU-test-b200"}, "gpu_family": "B200",
                    "cuda_arch": "sm_100", "nvcc_path": "/test/bin/nvcc"}
        commands = []

        def execute(command, **kwargs):
            commands.append(command)
            if "--external-probe" in command:
                kwargs["stdout"].write('{"total_sm_count": 148}')
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "copy-only"
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(module, "preflight", return_value=metadata), \
                    patch.object(module.subprocess, "run", side_effect=execute):
                self.assertEqual(module.main(["--output-dir", str(output), "--modes", "4",
                                              "--blocks-per-sm", "24", "--tma-stages", "2",
                                              "--tma-copy-only"]), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["arguments"]["tma_copy_only"])
            validation = [c for c in commands if "--validate-only" in c]
            self.assertEqual(len(validation), 6)
            for command in validation:
                self.assertEqual("--tma-copy-only" in command, command[-1] == "4")
            timed = [c for c in commands if "--require-dram" in c]
            self.assertEqual(len(timed), 1)
            self.assertIn("--tma-copy-only", timed[0])

    def test_h200_build_uses_detected_arch_and_preserves_sources(self):
        spec = importlib.util.spec_from_file_location("runpod_build", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata = {"gpu": {"uuid": "GPU-test-h200"}, "gpu_family": "H200",
                    "cuda_arch": "sm_90", "nvcc_path": "/test/bin/nvcc"}
        commands = []

        def execute(command, **kwargs):
            commands.append((command, kwargs["env"], kwargs["cwd"]))
            if "--external-probe" in command:
                kwargs["stdout"].write('{"total_sm_count": 132}')
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "h200-run"
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(module, "preflight", return_value=metadata), \
                    patch.object(module.subprocess, "run", side_effect=execute):
                self.assertEqual(module.main(["--output-dir", str(output), "--modes", "4",
                                              "--blocks-per-sm", "8", "--tma-stages", "4"]), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["cuda_arch"], "sm_90")
            self.assertIn("run_h200_runpod.py", manifest["source_sha256"])
            build, environment, working_dir = commands[0]
            self.assertIn("CUDA_ARCH=sm_90", build)
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-test-h200")
            self.assertEqual(working_dir, output / "source")
            self.assertTrue((working_dir / "run_h200_runpod.py").is_file())
