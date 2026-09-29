import csv
import json
import tempfile
import unittest
from pathlib import Path


class RawMaskPlanningTests(unittest.TestCase):
    def test_raw_mask_commands_use_hex_mask_without_logical_tpc_validation(self):
        from run_raw_mask_sweep import build_probe_command, build_raw_mask_command

        probe = build_probe_command(
            nvtaskset=Path("/opt/libsmctrl/nvtaskset"),
            benchmark=Path("/work/green_ctx_bw_bench"),
            enabled_bits=(0, 3, 6, 9),
            gpu_id=0,
        )
        timed = build_raw_mask_command(
            nvtaskset=Path("/opt/libsmctrl/nvtaskset"),
            benchmark=Path("/work/green_ctx_bw_bench"),
            enabled_bits=(0, 3, 6, 9),
            buffer_mb=1024,
            iterations=30,
            gpu_id=0,
            trials=5,
            load_mode=1,
        )

        self.assertEqual(
            probe,
            [
                "/opt/libsmctrl/nvtaskset",
                "0x249",
                "/work/green_ctx_bw_bench",
                "--external-probe",
                "0",
            ],
        )
        self.assertEqual(timed[0:3], [
            "/opt/libsmctrl/nvtaskset", "0x249", "/work/green_ctx_bw_bench"
        ])

    def test_classify_effective_bits_uses_leave_one_out_sm_delta(self):
        from run_raw_mask_sweep import classify_effective_bits

        omitted_counts = {0: 6, 1: 8, 2: 7, 3: 6, 4: 8}

        groups = classify_effective_bits(full_sm_count=8, omitted_counts=omitted_counts)

        self.assertEqual(groups, {1: (2,), 2: (0, 3)})

    def test_classify_effective_bits_rejects_impossible_delta(self):
        from run_raw_mask_sweep import classify_effective_bits

        with self.assertRaisesRegex(ValueError, "changed the visible SM count by 3"):
            classify_effective_bits(full_sm_count=8, omitted_counts={0: 5})

    def test_select_uniform_effective_bits_prefers_largest_population(self):
        from run_raw_mask_sweep import select_uniform_effective_bits

        bits, sms_per_bit = select_uniform_effective_bits({1: (7,), 2: (0, 3, 5)})

        self.assertEqual(bits, (0, 3, 5))
        self.assertEqual(sms_per_bit, 2)

    def test_generate_random_masks_is_unique_and_deterministic(self):
        from run_raw_mask_sweep import generate_random_masks

        first = generate_random_masks((0, 2, 4, 6, 8, 10), 3, 5, seed=8228)
        second = generate_random_masks((0, 2, 4, 6, 8, 10), 3, 5, seed=8228)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all(len(mask) == 3 for mask in first))

    def test_generate_random_masks_rejects_more_than_combinations(self):
        from run_raw_mask_sweep import generate_random_masks

        with self.assertRaisesRegex(ValueError, "only 3 unique masks exist"):
            generate_random_masks((0, 1, 2), 2, 4, seed=1)

    def test_generate_packed_masks_uses_adjacent_effective_slots(self):
        from run_raw_mask_sweep import generate_packed_masks

        masks = generate_packed_masks(
            (0, 2, 4, 6, 8, 10), 3, 3, seed=8228
        )

        self.assertEqual(len(masks), 3)
        self.assertEqual(len(set(masks)), 3)
        positions = {bit: index for index, bit in enumerate((0, 2, 4, 6, 8, 10))}
        for mask in masks:
            self.assertEqual(
                [positions[bit] for bit in mask],
                list(range(positions[mask[0]], positions[mask[0]] + 3)),
            )

    def test_generate_scattered_masks_maximizes_slot_separation(self):
        from run_raw_mask_sweep import generate_scattered_masks

        masks = generate_scattered_masks(tuple(range(12)), 3, 3, seed=8228)

        self.assertEqual(len(masks), 3)
        self.assertEqual(len(set(masks)), 3)
        for mask in masks:
            gaps = [right - left for left, right in zip(mask, mask[1:])]
            self.assertGreaterEqual(min(gaps), 5)

    def test_generate_scattered_masks_is_deterministic(self):
        from run_raw_mask_sweep import generate_scattered_masks

        first = generate_scattered_masks(tuple(range(20)), 4, 5, seed=8228)
        second = generate_scattered_masks(tuple(range(20)), 4, 5, seed=8228)

        self.assertEqual(first, second)

    def test_summarize_mask_rows_separates_between_and_within_mask_variation(self):
        from run_raw_mask_sweep import summarize_mask_rows

        rows = [
            {"tpc_count": 2, "mask_id": "m000", "tpc_list": "0,2", "repetition": 0,
             "bandwidth_GBps": 100.0, "observed_sm_count": 4},
            {"tpc_count": 2, "mask_id": "m000", "tpc_list": "0,2", "repetition": 1,
             "bandwidth_GBps": 102.0, "observed_sm_count": 4},
            {"tpc_count": 2, "mask_id": "m001", "tpc_list": "4,6", "repetition": 0,
             "bandwidth_GBps": 120.0, "observed_sm_count": 4},
            {"tpc_count": 2, "mask_id": "m001", "tpc_list": "4,6", "repetition": 1,
             "bandwidth_GBps": 122.0, "observed_sm_count": 4},
        ]

        mask_rows, summary_rows = summarize_mask_rows(rows, expected_sms_per_bit=2)

        self.assertEqual([row["median_bandwidth_GBps"] for row in mask_rows], [101.0, 121.0])
        self.assertEqual(summary_rows[0]["slowest_mask"], "m000")
        self.assertEqual(summary_rows[0]["fastest_mask"], "m001")
        self.assertAlmostEqual(summary_rows[0]["between_mask_spread_pct"], 20.0 / 111.0 * 100.0)
        self.assertLess(summary_rows[0]["median_within_mask_cv_pct"], 1.1)

    def test_summarize_mask_rows_keeps_layouts_separate(self):
        from run_raw_mask_sweep import summarize_mask_rows

        rows = [
            {"tpc_count": 2, "mask_layout": "packed", "mask_id": "packed000",
             "tpc_list": "0,1", "repetition": 0, "bandwidth_GBps": 80.0,
             "observed_sm_count": 4},
            {"tpc_count": 2, "mask_layout": "scattered", "mask_id": "scattered000",
             "tpc_list": "0,9", "repetition": 0, "bandwidth_GBps": 100.0,
             "observed_sm_count": 4},
        ]

        mask_rows, summary_rows = summarize_mask_rows(rows, expected_sms_per_bit=2)

        self.assertEqual(
            {(row["mask_layout"], row["median_bandwidth_GBps"]) for row in mask_rows},
            {("packed", 80.0), ("scattered", 100.0)},
        )
        self.assertEqual(
            {(row["mask_layout"], row["median_bandwidth_GBps"]) for row in summary_rows},
            {("packed", 80.0), ("scattered", 100.0)},
        )

    def test_compare_layout_summaries_reports_scattered_recovery(self):
        from run_raw_mask_sweep import compare_layout_summaries

        comparisons = compare_layout_summaries(
            [
                {"tpc_count": 2, "mask_layout": "packed", "median_bandwidth_GBps": 80.0},
                {"tpc_count": 2, "mask_layout": "scattered", "median_bandwidth_GBps": 100.0},
            ]
        )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["scattered_minus_packed_GBps"], 20.0)
        self.assertEqual(comparisons[0]["scattered_over_packed"], 1.25)
        self.assertEqual(comparisons[0]["scattered_gain_pct"], 25.0)


class RawMaskRunnerIntegrationTests(unittest.TestCase):
    def test_main_probes_bits_and_runs_random_fixed_cardinality_masks(self):
        from run_raw_mask_sweep import main

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
                "env['FAKE_TPCS'] = ','.join(str(bit) for bit in range(128) "
                "if mask & (1 << bit))\n"
                "os.execvpe(sys.argv[2], sys.argv[2:], env)\n"
            )
            nvtaskset.chmod(0o755)

            benchmark = temp_dir / "fake_benchmark.py"
            benchmark.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "bits = [int(v) for v in os.environ['FAKE_TPCS'].split(',')]\n"
                "effective = {0, 2, 4, 5}\n"
                "visible = len(set(bits) & effective) * 2\n"
                "if sys.argv[1] == '--external-probe':\n"
                "    print(json.dumps({'kind': 'external_probe', 'observed_sm_count': visible, "
                "'total_sm_count': 8, 'observed_sm_ids': list(range(visible))}))\n"
                "else:\n"
                "    bw = 100.0 + sum(set(bits) & effective)\n"
                "    print(json.dumps({'kind': 'external_single', 'bandwidth_GBps': bw, "
                "'observed_sm_count': visible, 'observed_sm_ids': list(range(visible))}))\n"
            )
            benchmark.chmod(0o755)

            prefix = temp_dir / "rawmask"
            result = main(
                [
                    "--libsmctrl-dir", str(libsmctrl_dir),
                    "--benchmark", str(benchmark),
                    "--candidate-bits", "6",
                    "--tpc-counts", "2",
                    "--masks-per-count", "3",
                    "--repetitions", "2",
                    "--output-prefix", str(prefix),
                ]
            )

            self.assertEqual(result, 0)
            with (temp_dir / "rawmask_probe.csv").open(newline="") as handle:
                probe_rows = list(csv.DictReader(handle))
            self.assertEqual(len(probe_rows), 6)
            self.assertEqual(
                {row["bit"] for row in probe_rows if row["sm_delta"] == "2"},
                {"0", "2", "4", "5"},
            )

            with (temp_dir / "rawmask_raw.csv").open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            self.assertEqual(len(raw_rows), 6)
            self.assertEqual({row["observed_sm_count"] for row in raw_rows}, {"4"})

            with (temp_dir / "rawmask_summary.csv").open(newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["tpc_count"], "2")

            metadata = json.loads((temp_dir / "rawmask_metadata.json").read_text())
            self.assertEqual(metadata["effective_bits"], [0, 2, 4, 5])
            self.assertEqual(metadata["sms_per_bit"], 2)

    def test_main_runs_packed_and_scattered_masks_as_separate_layouts(self):
        from run_raw_mask_sweep import main

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
                "env['FAKE_TPCS'] = ','.join(str(bit) for bit in range(128) "
                "if mask & (1 << bit))\n"
                "os.execvpe(sys.argv[2], sys.argv[2:], env)\n"
            )
            nvtaskset.chmod(0o755)
            benchmark = temp_dir / "fake_benchmark.py"
            benchmark.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "bits = [int(v) for v in os.environ['FAKE_TPCS'].split(',')]\n"
                "visible = len(set(bits) & set(range(8))) * 2\n"
                "kind = 'external_probe' if sys.argv[1] == '--external-probe' else 'external_single'\n"
                "print(json.dumps({'kind': kind, 'bandwidth_GBps': 100.0, "
                "'observed_sm_count': visible, 'total_sm_count': 16, "
                "'observed_sm_ids': list(range(visible))}))\n"
            )
            benchmark.chmod(0o755)

            prefix = temp_dir / "layouts"
            result = main(
                [
                    "--libsmctrl-dir", str(libsmctrl_dir),
                    "--benchmark", str(benchmark),
                    "--candidate-bits", "8",
                    "--tpc-counts", "2",
                    "--mask-layouts", "packed,scattered",
                    "--masks-per-count", "2",
                    "--repetitions", "1",
                    "--output-prefix", str(prefix),
                ]
            )

            self.assertEqual(result, 0)
            with (temp_dir / "layouts_raw.csv").open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            self.assertEqual(len(raw_rows), 4)
            self.assertEqual(
                {row["mask_layout"] for row in raw_rows}, {"packed", "scattered"}
            )
            with (temp_dir / "layouts_summary.csv").open(newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 2)
            with (temp_dir / "layouts_comparison.csv").open(newline="") as handle:
                comparison_rows = list(csv.DictReader(handle))
            self.assertEqual(comparison_rows[0]["scattered_over_packed"], "1.0")


if __name__ == "__main__":
    unittest.main()
