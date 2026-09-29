import json
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GpcTopologyTests(unittest.TestCase):
    def test_load_topology_json_reuses_saved_physical_mapping(self):
        from run_gpc_sweep import load_topology_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topology.json"
            path.write_text(
                json.dumps(
                    {
                        "topology": {
                            "0": [0, 7, 14],
                            "1": [1, 8, 15],
                            "2": [2, 9, 16],
                        },
                        "raw_libsmctrl_output": "verified GPU1 topology",
                    }
                )
            )

            topology, provenance = load_topology_json(path)

        self.assertEqual(
            topology,
            {0: [0, 7, 14], 1: [1, 8, 15], 2: [2, 9, 16]},
        )
        self.assertEqual(provenance, "verified GPU1 topology")

    def test_load_topology_json_rejects_overlapping_tpcs(self):
        from run_gpc_sweep import load_topology_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topology.json"
            path.write_text(
                json.dumps({"topology": {"0": [0, 1], "1": [1, 2]}})
            )

            with self.assertRaisesRegex(
                ValueError, "TPC 1 appears in multiple GPCs"
            ):
                load_topology_json(path)

    def test_parse_gpc_info_extracts_tpc_ids_from_masks(self):
        from run_gpc_sweep import parse_gpc_info

        output = """
libsmctrl_test_gpc_info: GPU0 has 3 enabled GPCs.
libsmctrl_test_gpc_info: Mask of 3 TPCs associated with GPC 0: 0x0000000000000007
libsmctrl_test_gpc_info: Mask of 2 TPCs associated with GPC 1: 0x0000000000000018
libsmctrl_test_gpc_info: Mask of 1 TPCs associated with GPC 2: 0x0000000000000020
libsmctrl_test_gpc_info: Total of 6 enabled TPCs.
"""

        self.assertEqual(parse_gpc_info(output), {0: [0, 1, 2], 1: [3, 4], 2: [5]})

    def test_parse_gpc_info_rejects_overlapping_tpc_masks(self):
        from run_gpc_sweep import parse_gpc_info

        output = """
tool: GPU0 has 2 enabled GPCs.
tool: Mask of 2 TPCs associated with GPC 0: 0x3
tool: Mask of 2 TPCs associated with GPC 1: 0x6
tool: Total of 4 enabled TPCs.
"""

        with self.assertRaisesRegex(ValueError, "TPC 1 appears in multiple GPCs"):
            parse_gpc_info(output)

    def test_build_placements_minimizes_and_maximizes_gpc_span(self):
        from run_gpc_sweep import build_placements

        topology = {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8],
            3: [9, 10, 11],
        }

        packed, scattered = build_placements(topology, 4)

        self.assertEqual(packed.tpcs, (0, 1, 2, 3))
        self.assertEqual(packed.gpcs, (0, 1))
        self.assertEqual(scattered.tpcs, (0, 3, 6, 9))
        self.assertEqual(scattered.gpcs, (0, 1, 2, 3))

    def test_build_placements_uses_largest_gpcs_for_uneven_topology(self):
        from run_gpc_sweep import build_placements

        topology = {0: [0, 1], 1: [2, 3, 4], 2: [5]}

        packed, scattered = build_placements(topology, 4)

        self.assertEqual(packed.tpcs, (0, 2, 3, 4))
        self.assertEqual(packed.gpcs, (0, 1))
        self.assertEqual(scattered.tpcs, (0, 1, 2, 5))
        self.assertEqual(scattered.gpcs, (0, 1, 2))

    def test_build_placements_rejects_count_above_available_tpcs(self):
        from run_gpc_sweep import build_placements

        with self.assertRaisesRegex(ValueError, "only 3 TPCs are available"):
            build_placements({0: [0, 1], 1: [2]}, 4)


class CommandTests(unittest.TestCase):
    def test_build_command_wraps_external_single_run_with_raw_mask(self):
        from run_gpc_sweep import build_command

        command = build_command(
            nvtaskset=Path("/opt/libsmctrl/nvtaskset"),
            benchmark=Path("/work/green_ctx_bw_bench"),
            tpcs=(0, 3, 6, 9),
            buffer_mb=1024,
            iterations=30,
            gpu_id=0,
            trials=7,
            load_mode=1,
        )

        self.assertEqual(
            command,
            [
                "/opt/libsmctrl/nvtaskset",
                "0x249",
                "/work/green_ctx_bw_bench",
                "--external-single",
                "1024",
                "30",
                "0",
                "7",
                "1",
            ],
        )

    def test_parse_benchmark_json_requires_external_single_record(self):
        from run_gpc_sweep import parse_benchmark_json

        stdout = "diagnostic\n" + json.dumps(
            {
                "kind": "external_single",
                "bandwidth_GBps": 6100.25,
                "observed_sm_count": 16,
                "observed_sm_ids": [0, 1, 8, 9],
            }
        )

        result = parse_benchmark_json(stdout)
        self.assertEqual(result["bandwidth_GBps"], 6100.25)
        self.assertEqual(result["observed_sm_count"], 16)

        with self.assertRaisesRegex(ValueError, "external_single JSON record"):
            parse_benchmark_json('{"kind":"something_else"}\n')


class ExperimentPlanningTests(unittest.TestCase):
    def test_auto_tpc_counts_cover_gpc_steps_and_forty_to_fifty_percent(self):
        from run_gpc_sweep import auto_tpc_counts

        topology = {
            gpc: list(range(gpc * 9, (gpc + 1) * 9))
            for gpc in range(8)
        }

        self.assertEqual(auto_tpc_counts(topology), (9, 18, 27, 29, 36))

    def test_summarize_rows_pairs_equal_sm_counts_and_reports_gain(self):
        from run_gpc_sweep import summarize_rows

        rows = [
            {
                "tpc_count": 4,
                "placement": "packed",
                "repetition": 0,
                "bandwidth_GBps": 100.0,
                "observed_sm_count": 8,
            },
            {
                "tpc_count": 4,
                "placement": "scattered",
                "repetition": 0,
                "bandwidth_GBps": 110.0,
                "observed_sm_count": 8,
            },
            {
                "tpc_count": 4,
                "placement": "packed",
                "repetition": 1,
                "bandwidth_GBps": 102.0,
                "observed_sm_count": 8,
            },
            {
                "tpc_count": 4,
                "placement": "scattered",
                "repetition": 1,
                "bandwidth_GBps": 114.0,
                "observed_sm_count": 8,
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["tpc_count"], 4)
        self.assertEqual(summary[0]["paired_repetitions"], 2)
        self.assertEqual(summary[0]["median_packed_GBps"], 101.0)
        self.assertEqual(summary[0]["median_scattered_GBps"], 112.0)
        self.assertAlmostEqual(summary[0]["scattered_over_packed"], 112.0 / 101.0)
        self.assertEqual(summary[0]["scattered_wins"], 2)

    def test_summarize_rows_rejects_unequal_observed_sm_counts(self):
        from run_gpc_sweep import summarize_rows

        rows = [
            {
                "tpc_count": 4,
                "placement": "packed",
                "repetition": 0,
                "bandwidth_GBps": 100.0,
                "observed_sm_count": 7,
            },
            {
                "tpc_count": 4,
                "placement": "scattered",
                "repetition": 0,
                "bandwidth_GBps": 110.0,
                "observed_sm_count": 8,
            },
        ]

        with self.assertRaisesRegex(ValueError, "observed 7 versus 8 SMs"):
            summarize_rows(rows)

    def test_summarize_rows_rejects_impossible_sm_count_for_tpc_mask(self):
        from run_gpc_sweep import summarize_rows

        rows = [
            {
                "tpc_count": 4,
                "placement": placement,
                "repetition": 0,
                "bandwidth_GBps": 100.0,
                "observed_sm_count": 3,
            }
            for placement in ("packed", "scattered")
        ]

        with self.assertRaisesRegex(ValueError, "expected between 4 and 8"):
            summarize_rows(rows)

    def test_summarize_rows_rejects_missing_pair(self):
        from run_gpc_sweep import summarize_rows

        rows = [
            {
                "tpc_count": 4,
                "placement": "packed",
                "repetition": 0,
                "bandwidth_GBps": 100.0,
                "observed_sm_count": 8,
            }
        ]

        with self.assertRaisesRegex(ValueError, "missing packed/scattered pair"):
            summarize_rows(rows)


class CliContractTests(unittest.TestCase):
    def test_help_documents_external_single_mode_without_requiring_a_gpu(self):
        binary = ROOT / "green_ctx_bw_bench"
        if not binary.exists():
            self.skipTest("build green_ctx_bw_bench before running the CLI contract test")

        result = subprocess.run(
            [str(binary), "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--external-single", result.stdout)

    def test_external_single_help_describes_machine_readable_output(self):
        binary = ROOT / "green_ctx_bw_bench"
        if not binary.exists():
            self.skipTest("build green_ctx_bw_bench before running the CLI contract test")

        result = subprocess.run(
            [str(binary), "--external-single", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("external_single JSON", result.stdout)
        self.assertIn("observed_sm_ids", result.stdout)

    def test_external_probe_help_describes_residency_only_mode(self):
        binary = ROOT / "green_ctx_bw_bench"
        if not binary.exists():
            self.skipTest("build green_ctx_bw_bench before running the CLI contract test")

        result = subprocess.run(
            [str(binary), "--external-probe", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("external_probe JSON", result.stdout)

    def test_runner_help_documents_required_libsmctrl_directory(self):
        result = subprocess.run(
            ["python3", str(ROOT / "run_gpc_sweep.py"), "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--libsmctrl-dir", result.stdout)
        self.assertIn("--topology-json", result.stdout)
        self.assertIn("--tpc-counts", result.stdout)


class RunnerIntegrationTests(unittest.TestCase):
    def test_main_can_capture_topology_without_starting_benchmark(self):
        from run_gpc_sweep import main

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            libsmctrl_dir = temp_dir / "libsmctrl"
            libsmctrl_dir.mkdir()

            topology_tool = libsmctrl_dir / "libsmctrl_test_gpc_info"
            topology_tool.write_text(
                "#!/bin/sh\n"
                "echo 'tool: GPU1 has 2 enabled GPCs.'\n"
                "echo 'tool: Mask of 2 TPCs associated with GPC 0: 0x05'\n"
                "echo 'tool: Mask of 2 TPCs associated with GPC 1: 0x0a'\n"
                "echo 'tool: Total of 4 enabled TPCs.'\n"
            )
            topology_tool.chmod(0o755)

            output = temp_dir / "gpu1.json"
            result = main(
                [
                    "--libsmctrl-dir",
                    str(libsmctrl_dir),
                    "--capture-topology-json",
                    str(output),
                    "--gpu-id",
                    "1",
                ]
            )

            self.assertEqual(result, 0)
            document = json.loads(output.read_text())
            self.assertEqual(document["topology"], {"0": [0, 2], "1": [1, 3]})
            self.assertIn("GPU1 has 2 enabled GPCs", document["raw_libsmctrl_output"])

    def test_main_runs_paired_masks_and_writes_raw_and_summary_csv(self):
        from run_gpc_sweep import main

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            libsmctrl_dir = temp_dir / "libsmctrl"
            libsmctrl_dir.mkdir()

            topology_tool = libsmctrl_dir / "libsmctrl_test_gpc_info"
            topology_tool.write_text(
                "#!/bin/sh\n"
                "echo 'tool: GPU0 has 4 enabled GPCs.'\n"
                "echo 'tool: Mask of 2 TPCs associated with GPC 0: 0x03'\n"
                "echo 'tool: Mask of 2 TPCs associated with GPC 1: 0x0c'\n"
                "echo 'tool: Mask of 2 TPCs associated with GPC 2: 0x30'\n"
                "echo 'tool: Mask of 2 TPCs associated with GPC 3: 0xc0'\n"
                "echo 'tool: Total of 8 enabled TPCs.'\n"
            )
            topology_tool.chmod(0o755)

            nvtaskset = libsmctrl_dir / "nvtaskset"
            nvtaskset.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "mask = int(sys.argv[1], 16)\n"
                "env = os.environ.copy()\n"
                "env['FAKE_TPCS'] = ','.join(\n"
                "    str(bit) for bit in range(128) if mask & (1 << bit)\n"
                ")\n"
                "os.execvpe(sys.argv[2], sys.argv[2:], env)\n"
            )
            nvtaskset.chmod(0o755)

            benchmark = temp_dir / "fake_benchmark.py"
            benchmark.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "tpcs = [int(value) for value in os.environ['FAKE_TPCS'].split(',')]\n"
                "scattered = tpcs == [0, 2, 4, 6]\n"
                "print(json.dumps({'kind': 'external_single', "
                "'bandwidth_GBps': 120.0 if scattered else 100.0, "
                "'observed_sm_count': len(tpcs) * 2, "
                "'observed_sm_ids': list(range(len(tpcs) * 2))}))\n"
            )
            benchmark.chmod(0o755)

            output_prefix = temp_dir / "result"
            result = main(
                [
                    "--libsmctrl-dir",
                    str(libsmctrl_dir),
                    "--benchmark",
                    str(benchmark),
                    "--tpc-counts",
                    "4",
                    "--repetitions",
                    "2",
                    "--output-prefix",
                    str(output_prefix),
                ]
            )

            self.assertEqual(result, 0)
            with (temp_dir / "result_raw.csv").open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            self.assertEqual(len(raw_rows), 4)
            self.assertEqual({row["placement"] for row in raw_rows}, {"packed", "scattered"})
            self.assertEqual({row["observed_sm_count"] for row in raw_rows}, {"8"})

            with (temp_dir / "result_summary.csv").open(newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["tpc_count"], "4")
            self.assertEqual(summary_rows[0]["scattered_over_packed"], "1.2")
            self.assertEqual(summary_rows[0]["scattered_wins"], "2")


    def test_main_uses_saved_topology_without_live_topology_tool(self):
        from run_gpc_sweep import main

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            libsmctrl_dir = temp_dir / "libsmctrl"
            libsmctrl_dir.mkdir()

            nvtaskset = libsmctrl_dir / "nvtaskset"
            nvtaskset.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "mask = int(sys.argv[1], 16)\n"
                "env = os.environ.copy()\n"
                "env['FAKE_TPCS'] = ','.join(\n"
                "    str(bit) for bit in range(128) if mask & (1 << bit)\n"
                ")\n"
                "os.execvpe(sys.argv[2], sys.argv[2:], env)\n"
            )
            nvtaskset.chmod(0o755)

            benchmark = temp_dir / "fake_benchmark.py"
            benchmark.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "tpcs = [int(value) for value in os.environ['FAKE_TPCS'].split(',')]\n"
                "print(json.dumps({'kind': 'external_single', "
                "'bandwidth_GBps': 100.0, "
                "'observed_sm_count': len(tpcs) * 2, "
                "'observed_sm_ids': list(range(len(tpcs) * 2))}))\n"
            )
            benchmark.chmod(0o755)

            topology_path = temp_dir / "gpu1_topology.json"
            topology_path.write_text(
                json.dumps(
                    {
                        "topology": {
                            "0": [0, 4],
                            "1": [1, 5],
                            "2": [2, 6],
                            "3": [3, 7],
                        },
                        "raw_libsmctrl_output": "saved GPU1 topology",
                    }
                )
            )

            output_prefix = temp_dir / "saved"
            result = main(
                [
                    "--libsmctrl-dir",
                    str(libsmctrl_dir),
                    "--topology-json",
                    str(topology_path),
                    "--benchmark",
                    str(benchmark),
                    "--tpc-counts",
                    "4",
                    "--repetitions",
                    "1",
                    "--output-prefix",
                    str(output_prefix),
                ]
            )

            self.assertEqual(result, 0)
if __name__ == "__main__":
    unittest.main()
