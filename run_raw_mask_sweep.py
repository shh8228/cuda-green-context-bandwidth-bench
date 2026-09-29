#!/usr/bin/env python3
"""Measure fixed-size raw TPC masks without requiring a GPC topology map."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from run_gpc_sweep import (
    _prefixed_output_path,
    _subprocess_environment,
    _write_csv,
    parse_benchmark_json,
)


def classify_effective_bits(
    *, full_sm_count: int, omitted_counts: Mapping[int, int]
) -> dict[int, tuple[int, ...]]:
    """Group raw mask bits by the SM-count decrease when each bit is omitted."""
    groups: dict[int, list[int]] = {1: [], 2: []}
    for bit, omitted_count in sorted(omitted_counts.items()):
        delta = full_sm_count - omitted_count
        if delta == 0:
            continue
        if delta not in groups:
            raise ValueError(
                f"omitting raw mask bit {bit} changed the visible SM count by {delta}; "
                "expected 0, 1, or 2"
            )
        groups[delta].append(bit)
    return {delta: tuple(bits) for delta, bits in groups.items() if bits}


def select_uniform_effective_bits(
    groups: Mapping[int, Sequence[int]]
) -> tuple[tuple[int, ...], int]:
    """Select the largest population having a uniform SM count per mask bit."""
    if not groups:
        raise ValueError("no effective raw TPC mask bits were discovered")
    sms_per_bit, bits = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
    return tuple(sorted(bits)), int(sms_per_bit)


def generate_random_masks(
    effective_bits: Sequence[int],
    tpc_count: int,
    mask_count: int,
    *,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Generate deterministic unique fixed-cardinality masks."""
    bits = tuple(sorted(effective_bits))
    if not 0 < tpc_count <= len(bits):
        raise ValueError(
            f"TPC count must be between 1 and {len(bits)}, got {tpc_count}"
        )
    combinations = math.comb(len(bits), tpc_count)
    if mask_count > combinations:
        raise ValueError(
            f"requested {mask_count} masks, but only {combinations} unique masks exist"
        )

    rng = random.Random(seed)
    masks: set[tuple[int, ...]] = set()
    while len(masks) < mask_count:
        masks.add(tuple(sorted(rng.sample(bits, tpc_count))))
    return tuple(sorted(masks))


def _validate_mask_request(
    effective_bits: Sequence[int], tpc_count: int, mask_count: int
) -> tuple[int, ...]:
    bits = tuple(sorted(effective_bits))
    if not 0 < tpc_count <= len(bits):
        raise ValueError(
            f"TPC count must be between 1 and {len(bits)}, got {tpc_count}"
        )
    if mask_count <= 0:
        raise ValueError("mask count must be positive")
    return bits


def generate_packed_masks(
    effective_bits: Sequence[int],
    tpc_count: int,
    mask_count: int,
    *,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Select masks made from adjacent positions in the effective-bit list."""
    bits = _validate_mask_request(effective_bits, tpc_count, mask_count)
    windows = tuple(
        bits[start : start + tpc_count]
        for start in range(len(bits) - tpc_count + 1)
    )
    if mask_count > len(windows):
        raise ValueError(
            f"requested {mask_count} packed masks, but only {len(windows)} "
            "unique packed windows exist"
        )
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(windows, mask_count)))


def generate_scattered_masks(
    effective_bits: Sequence[int],
    tpc_count: int,
    mask_count: int,
    *,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Select masks whose positions are as evenly and widely spaced as possible."""
    bits = _validate_mask_request(effective_bits, tpc_count, mask_count)
    if tpc_count == 1:
        if mask_count > len(bits):
            raise ValueError(
                f"requested {mask_count} scattered masks, but only {len(bits)} "
                "unique one-bit masks exist"
            )
        rng = random.Random(seed)
        return tuple(sorted((bits[index],) for index in rng.sample(range(len(bits)), mask_count)))

    position_masks: set[tuple[int, ...]] = set()
    for first in range(len(bits)):
        for last in range(first + tpc_count - 1, len(bits)):
            span = last - first
            positions = tuple(
                round(first + index * span / (tpc_count - 1))
                for index in range(tpc_count)
            )
            if len(set(positions)) == tpc_count:
                position_masks.add(positions)
    if mask_count > len(position_masks):
        raise ValueError(
            f"requested {mask_count} scattered masks, but only "
            f"{len(position_masks)} unique evenly spaced masks exist"
        )

    candidates = list(position_masks)
    random.Random(seed).shuffle(candidates)

    def dispersion_score(positions: tuple[int, ...]) -> tuple[int, int, int]:
        gaps = tuple(
            right - left for left, right in zip(positions, positions[1:])
        )
        return min(gaps), positions[-1] - positions[0], -(max(gaps) - min(gaps))

    candidates.sort(key=dispersion_score, reverse=True)
    selected = candidates[:mask_count]
    return tuple(sorted(tuple(bits[position] for position in mask) for mask in selected))


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = statistics.mean(values)
    return 0.0 if mean == 0.0 else statistics.pstdev(values) / mean * 100.0


def summarize_mask_rows(
    rows: Sequence[Mapping[str, object]], *, expected_sms_per_bit: int
) -> tuple[list[dict], list[dict]]:
    """Summarize repeat noise per mask and bandwidth spread across masks."""
    grouped: dict[tuple[int, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            int(row["tpc_count"]),
            str(row.get("mask_layout", "random")),
            str(row["mask_id"]),
        )
        grouped.setdefault(key, []).append(row)

    mask_rows: list[dict] = []
    for (tpc_count, mask_layout, mask_id), samples in sorted(grouped.items()):
        expected_sms = tpc_count * expected_sms_per_bit
        observed_counts = {int(sample["observed_sm_count"]) for sample in samples}
        if observed_counts != {expected_sms}:
            raise ValueError(
                f"mask {mask_id} at {tpc_count} TPCs observed SM counts "
                f"{sorted(observed_counts)}, expected {expected_sms}"
            )
        bandwidths = [float(sample["bandwidth_GBps"]) for sample in samples]
        mask_rows.append(
            {
                "tpc_count": tpc_count,
                "mask_layout": mask_layout,
                "mask_id": mask_id,
                "tpc_list": str(samples[0]["tpc_list"]),
                "observed_sm_count": expected_sms,
                "repetitions": len(samples),
                "median_bandwidth_GBps": statistics.median(bandwidths),
                "within_mask_cv_pct": _coefficient_of_variation(bandwidths),
                "min_bandwidth_GBps": min(bandwidths),
                "max_bandwidth_GBps": max(bandwidths),
            }
        )

    by_count: dict[tuple[int, str], list[dict]] = {}
    for row in mask_rows:
        key = (int(row["tpc_count"]), str(row["mask_layout"]))
        by_count.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (tpc_count, mask_layout), masks in sorted(by_count.items()):
        medians = [float(mask["median_bandwidth_GBps"]) for mask in masks]
        center = statistics.median(medians)
        slowest = min(masks, key=lambda mask: float(mask["median_bandwidth_GBps"]))
        fastest = max(masks, key=lambda mask: float(mask["median_bandwidth_GBps"]))
        summary_rows.append(
            {
                "tpc_count": tpc_count,
                "mask_layout": mask_layout,
                "observed_sm_count": tpc_count * expected_sms_per_bit,
                "mask_count": len(masks),
                "median_bandwidth_GBps": center,
                "min_mask_median_GBps": min(medians),
                "max_mask_median_GBps": max(medians),
                "between_mask_spread_pct": (max(medians) - min(medians)) / center * 100.0,
                "max_over_min": max(medians) / min(medians),
                "median_within_mask_cv_pct": statistics.median(
                    float(mask["within_mask_cv_pct"]) for mask in masks
                ),
                "slowest_mask": str(slowest["mask_id"]),
                "slowest_tpc_list": str(slowest["tpc_list"]),
                "fastest_mask": str(fastest["mask_id"]),
                "fastest_tpc_list": str(fastest["tpc_list"]),
            }
        )
    return mask_rows, summary_rows


def compare_layout_summaries(summary_rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Compare packed and scattered layout medians at each TPC count."""
    by_count: dict[int, dict[str, Mapping[str, object]]] = {}
    for row in summary_rows:
        by_count.setdefault(int(row["tpc_count"]), {})[str(row["mask_layout"])] = row

    comparisons: list[dict] = []
    for tpc_count, layouts in sorted(by_count.items()):
        if "packed" not in layouts or "scattered" not in layouts:
            continue
        packed = float(layouts["packed"]["median_bandwidth_GBps"])
        scattered = float(layouts["scattered"]["median_bandwidth_GBps"])
        comparisons.append(
            {
                "tpc_count": tpc_count,
                "observed_sm_count": int(layouts["packed"]["observed_sm_count"])
                if "observed_sm_count" in layouts["packed"]
                else "",
                "packed_median_GBps": packed,
                "scattered_median_GBps": scattered,
                "scattered_minus_packed_GBps": scattered - packed,
                "scattered_over_packed": scattered / packed,
                "scattered_gain_pct": (scattered - packed) / packed * 100.0,
            }
        )
    return comparisons


def build_probe_command(
    *, nvtaskset: Path, benchmark: Path, enabled_bits: Sequence[int], gpu_id: int
) -> list[str]:
    return [
        str(nvtaskset),
        _raw_enable_mask(enabled_bits),
        str(benchmark),
        "--external-probe",
        str(gpu_id),
    ]


def _raw_enable_mask(enabled_bits: Sequence[int]) -> str:
    mask = 0
    for bit in enabled_bits:
        if not 0 <= bit < 128:
            raise ValueError(f"raw mask bit must be between 0 and 127, got {bit}")
        mask |= 1 << bit
    if mask == 0:
        raise ValueError("raw enable mask cannot be empty")
    return hex(mask)


def build_raw_mask_command(
    *,
    nvtaskset: Path,
    benchmark: Path,
    enabled_bits: Sequence[int],
    buffer_mb: int,
    iterations: int,
    gpu_id: int,
    trials: int,
    load_mode: int,
) -> list[str]:
    return [
        str(nvtaskset),
        _raw_enable_mask(enabled_bits),
        str(benchmark),
        "--external-single",
        str(buffer_mb),
        str(iterations),
        str(gpu_id),
        str(trials),
        str(load_mode),
    ]


def _run_process(
    command: Sequence[str], environment: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            text=True,
            capture_output=True,
            env=dict(environment),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"command timed out after {timeout:g}s: {' '.join(command)}"
        ) from error
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}"
        )
    return result


def _parse_probe_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "external_probe":
            return record
    raise ValueError("benchmark output did not contain an external_probe JSON record")


def _parse_counts(value: str, effective_count: int) -> tuple[int, ...]:
    if value == "auto":
        fractions = (0.125, 0.25, 0.375, 0.40, 0.50)
        return tuple(sorted({max(1, math.ceil(effective_count * f)) for f in fractions}))
    try:
        counts = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    except ValueError as error:
        raise ValueError("--tpc-counts must be 'auto' or comma-separated integers") from error
    if not counts:
        raise ValueError("--tpc-counts did not contain any counts")
    if counts[0] <= 0 or counts[-1] > effective_count:
        raise ValueError(f"TPC counts must be between 1 and {effective_count}")
    return counts


def _parse_mask_layouts(value: str) -> tuple[str, ...]:
    layouts = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    allowed = {"random", "packed", "scattered"}
    invalid = set(layouts) - allowed
    if not layouts or invalid:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"--mask-layouts must contain only: {choices}")
    return layouts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover effective raw TPC mask bits and measure bandwidth variation "
            "across randomized equal-size masks without a GPC topology map."
        )
    )
    parser.add_argument("--libsmctrl-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("./green_ctx_bw_bench"))
    parser.add_argument("--candidate-bits", type=int, default=128)
    parser.add_argument("--probe-timeout", type=float, default=30.0)
    parser.add_argument("--tpc-counts", default="auto")
    parser.add_argument(
        "--mask-layouts",
        default="random",
        help="comma-separated mask layouts: random, packed, scattered",
    )
    parser.add_argument("--masks-per-count", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--buffer-mb", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--load-mode", type=int, default=1, choices=range(0, 6))
    parser.add_argument("--seed", type=int, default=8228)
    parser.add_argument("--output-prefix", type=Path, default=Path("b200_raw_mask_bw"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    for name in (
        "candidate_bits",
        "probe_timeout",
        "masks_per_count",
        "repetitions",
        "buffer_mb",
        "iterations",
        "trials",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.candidate_bits > 128:
        raise ValueError("libsmctrl supports at most 128 raw TPC mask bits")

    libsmctrl_dir = args.libsmctrl_dir.resolve()
    nvtaskset = libsmctrl_dir / "nvtaskset"
    benchmark = args.benchmark.resolve()
    if not nvtaskset.is_file():
        raise FileNotFoundError(f"nvtaskset not found: {nvtaskset}")
    if not benchmark.is_file():
        raise FileNotFoundError(f"benchmark not found: {benchmark}")
    environment = _subprocess_environment(libsmctrl_dir)

    probe_path = _prefixed_output_path(args.output_prefix, "_probe.csv")
    raw_path = _prefixed_output_path(args.output_prefix, "_raw.csv")
    masks_path = _prefixed_output_path(args.output_prefix, "_masks.csv")
    summary_path = _prefixed_output_path(args.output_prefix, "_summary.csv")
    comparison_path = _prefixed_output_path(args.output_prefix, "_comparison.csv")
    metadata_path = _prefixed_output_path(args.output_prefix, "_metadata.json")

    candidate_bits = tuple(range(args.candidate_bits))
    print(f"Probing all {args.candidate_bits} candidate bits together", file=sys.stderr)
    full_result = _run_process(
        build_probe_command(
            nvtaskset=nvtaskset,
            benchmark=benchmark,
            enabled_bits=candidate_bits,
            gpu_id=args.gpu_id,
        ),
        environment,
        args.probe_timeout,
    )
    full_probe = _parse_probe_json(full_result.stdout)
    full_sm_count = int(full_probe["observed_sm_count"])
    total_sm_count = int(full_probe["total_sm_count"])
    if full_sm_count != total_sm_count:
        raise ValueError(
            f"candidate bit range exposed {full_sm_count} of {total_sm_count} SMs; "
            "the raw mask search range is incomplete or masking is unsupported"
        )

    omitted_counts: dict[int, int] = {}
    probe_rows: list[dict] = []
    for index, bit in enumerate(candidate_bits):
        enabled = tuple(candidate for candidate in candidate_bits if candidate != bit)
        print(
            f"Leave-one-out probe {index + 1}/{len(candidate_bits)}: bit {bit}",
            file=sys.stderr,
        )
        result = _run_process(
            build_probe_command(
                nvtaskset=nvtaskset,
                benchmark=benchmark,
                enabled_bits=enabled,
                gpu_id=args.gpu_id,
            ),
            environment,
            args.probe_timeout,
        )
        record = _parse_probe_json(result.stdout)
        omitted_count = int(record["observed_sm_count"])
        omitted_counts[bit] = omitted_count
        probe_rows.append(
            {
                "bit": bit,
                "full_sm_count": full_sm_count,
                "omitted_sm_count": omitted_count,
                "sm_delta": full_sm_count - omitted_count,
            }
        )
        _write_csv(probe_path, probe_rows)

    groups = classify_effective_bits(
        full_sm_count=full_sm_count, omitted_counts=omitted_counts
    )
    effective_bits, sms_per_bit = select_uniform_effective_bits(groups)
    print(
        f"Using {len(effective_bits)} effective bits with {sms_per_bit} SM(s) per bit",
        file=sys.stderr,
    )
    counts = _parse_counts(args.tpc_counts, len(effective_bits))
    layouts = _parse_mask_layouts(args.mask_layouts)

    generators = {
        "random": generate_random_masks,
        "packed": generate_packed_masks,
        "scattered": generate_scattered_masks,
    }
    mask_specs: list[tuple[int, str, str, tuple[int, ...]]] = []
    for tpc_count in counts:
        for layout_index, layout in enumerate(layouts):
            masks = generators[layout](
                effective_bits,
                tpc_count,
                args.masks_per_count,
                seed=(
                    args.seed
                    ^ (tpc_count * 0x9E3779B1)
                    ^ (layout_index * 0x85EBCA6B)
                ),
            )
            for mask_index, mask in enumerate(masks):
                mask_specs.append(
                    (tpc_count, layout, f"{layout}{mask_index:03d}", mask)
                )

    run_specs = [
        (tpc_count, mask_layout, mask_id, mask, repetition)
        for tpc_count, mask_layout, mask_id, mask in mask_specs
        for repetition in range(args.repetitions)
    ]
    random.Random(args.seed).shuffle(run_specs)

    raw_rows: list[dict] = []
    for run_order, (tpc_count, mask_layout, mask_id, mask, repetition) in enumerate(
        run_specs
    ):
        print(
            f"Run {run_order + 1}/{len(run_specs)}: TPCs={tpc_count} "
            f"layout={mask_layout} mask={mask_id} "
            f"repetition={repetition + 1}/{args.repetitions}",
            file=sys.stderr,
        )
        command = build_raw_mask_command(
            nvtaskset=nvtaskset,
            benchmark=benchmark,
            enabled_bits=mask,
            buffer_mb=args.buffer_mb,
            iterations=args.iterations,
            gpu_id=args.gpu_id,
            trials=args.trials,
            load_mode=args.load_mode,
        )
        result = _run_process(command, environment, max(args.probe_timeout, 300.0))
        measurement = parse_benchmark_json(result.stdout)
        observed_sm_count = int(measurement["observed_sm_count"])
        expected_sm_count = tpc_count * sms_per_bit
        if observed_sm_count != expected_sm_count:
            raise ValueError(
                f"mask {mask_id} expected {expected_sm_count} SMs but observed "
                f"{observed_sm_count}; raw-mask behavior is not stable"
            )
        raw_rows.append(
            {
                "tpc_count": tpc_count,
                "mask_layout": mask_layout,
                "mask_id": mask_id,
                "tpc_list": ",".join(str(bit) for bit in mask),
                "repetition": repetition,
                "run_order": run_order,
                "bandwidth_GBps": float(measurement["bandwidth_GBps"]),
                "observed_sm_count": observed_sm_count,
                "observed_sm_ids": ",".join(
                    str(sm_id) for sm_id in measurement.get("observed_sm_ids", [])
                ),
            }
        )
        _write_csv(raw_path, raw_rows)

    mask_rows, summary_rows = summarize_mask_rows(
        raw_rows, expected_sms_per_bit=sms_per_bit
    )
    _write_csv(masks_path, mask_rows)
    _write_csv(summary_path, summary_rows)
    comparison_rows = compare_layout_summaries(summary_rows)
    if comparison_rows:
        _write_csv(comparison_path, comparison_rows)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "candidate_bits": args.candidate_bits,
                "full_sm_count": full_sm_count,
                "total_sm_count": total_sm_count,
                "effective_bit_groups": {
                    str(delta): list(bits) for delta, bits in groups.items()
                },
                "effective_bits": list(effective_bits),
                "sms_per_bit": sms_per_bit,
                "seed": args.seed,
                "load_mode": args.load_mode,
                "mask_layouts": list(layouts),
                "claim_scope": "raw-mask placement sensitivity; no GPC identity",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"Probe map: {probe_path}", file=sys.stderr)
    print(f"Raw runs:  {raw_path}", file=sys.stderr)
    print(f"Per-mask:  {masks_path}", file=sys.stderr)
    print(f"Summary:   {summary_path}", file=sys.stderr)
    if comparison_rows:
        print(f"Comparison:{comparison_path}", file=sys.stderr)
    print(f"Metadata:  {metadata_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
