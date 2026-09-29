#!/usr/bin/env python3
"""Run equal-size packed and scattered TPC placements through nvtaskset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple


_GPC_MASK_RE = re.compile(
    r"Mask of\s+(?P<count>\d+)\s+TPCs associated with GPC\s+"
    r"(?P<gpc>\d+):\s+0x(?P<mask>[0-9a-fA-F]+)"
)


@dataclass(frozen=True)
class Placement:
    tpcs: Tuple[int, ...]
    gpcs: Tuple[int, ...]


def _set_bits(mask: int) -> list[int]:
    return [bit for bit in range(mask.bit_length()) if mask & (1 << bit)]


def parse_gpc_info(output: str) -> Dict[int, list[int]]:
    """Parse libsmctrl_test_gpc_info output into GPC -> TPC IDs."""
    topology: Dict[int, list[int]] = {}
    owner: Dict[int, int] = {}
    for match in _GPC_MASK_RE.finditer(output):
        gpc = int(match.group("gpc"))
        declared_count = int(match.group("count"))
        tpcs = _set_bits(int(match.group("mask"), 16))
        if len(tpcs) != declared_count:
            raise ValueError(
                f"GPC {gpc} declares {declared_count} TPCs but its mask contains {len(tpcs)}"
            )
        for tpc in tpcs:
            if tpc in owner:
                raise ValueError(
                    f"TPC {tpc} appears in multiple GPCs: {owner[tpc]} and {gpc}"
                )
            owner[tpc] = gpc
        topology[gpc] = tpcs

    if not topology:
        raise ValueError("no GPC masks found in libsmctrl_test_gpc_info output")
    return dict(sorted(topology.items()))


def _validate_topology(raw_topology: object) -> Dict[int, list[int]]:
    if not isinstance(raw_topology, dict) or not raw_topology:
        raise ValueError("topology must be a non-empty GPC-to-TPC mapping")

    topology: Dict[int, list[int]] = {}
    owner: Dict[int, int] = {}
    for raw_gpc, raw_tpcs in raw_topology.items():
        try:
            gpc = int(raw_gpc)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid GPC ID {raw_gpc!r}") from error
        if gpc < 0 or gpc in topology:
            raise ValueError(f"invalid or duplicate GPC ID {gpc}")
        if not isinstance(raw_tpcs, list) or not raw_tpcs:
            raise ValueError(f"GPC {gpc} must contain at least one TPC")

        tpcs: list[int] = []
        for raw_tpc in raw_tpcs:
            if isinstance(raw_tpc, bool) or not isinstance(raw_tpc, int):
                raise ValueError(f"GPC {gpc} contains invalid TPC ID {raw_tpc!r}")
            tpc = raw_tpc
            if not 0 <= tpc < 128:
                raise ValueError(f"TPC ID {tpc} is outside nvtaskset's 128-bit mask")
            if tpc in owner:
                raise ValueError(
                    f"TPC {tpc} appears in multiple GPCs: {owner[tpc]} and {gpc}"
                )
            owner[tpc] = gpc
            tpcs.append(tpc)
        topology[gpc] = sorted(tpcs)
    return dict(sorted(topology.items()))


def load_topology_json(path: Path) -> tuple[Dict[int, list[int]], str]:
    """Load a topology captured while nvdebug was available."""
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or "topology" not in document:
        raise ValueError(f"{path} does not contain a topology mapping")
    topology = _validate_topology(document["topology"])
    provenance = str(document.get("raw_libsmctrl_output", ""))
    return topology, provenance


def _placement(topology: Mapping[int, Sequence[int]], selected: Iterable[int]) -> Placement:
    selected_set = set(selected)
    gpcs = tuple(
        sorted(gpc for gpc, tpcs in topology.items() if selected_set.intersection(tpcs))
    )
    return Placement(tuple(sorted(selected_set)), gpcs)


def build_placements(
    topology: Mapping[int, Sequence[int]], tpc_count: int
) -> tuple[Placement, Placement]:
    """Build equal-size placements with minimum and maximum GPC span."""
    total_tpcs = sum(len(tpcs) for tpcs in topology.values())
    if tpc_count <= 0:
        raise ValueError("TPC count must be positive")
    if tpc_count > total_tpcs:
        raise ValueError(
            f"requested {tpc_count} TPCs, but only {total_tpcs} TPCs are available"
        )

    packed_ids: list[int] = []
    for _, tpcs in sorted(topology.items(), key=lambda item: (-len(item[1]), item[0])):
        needed = tpc_count - len(packed_ids)
        packed_ids.extend(sorted(tpcs)[:needed])
        if len(packed_ids) == tpc_count:
            break

    scattered_ids: list[int] = []
    ordered_gpcs = [(gpc, sorted(tpcs)) for gpc, tpcs in sorted(topology.items())]
    depth = 0
    while len(scattered_ids) < tpc_count:
        made_progress = False
        for _, tpcs in ordered_gpcs:
            if depth < len(tpcs):
                scattered_ids.append(tpcs[depth])
                made_progress = True
                if len(scattered_ids) == tpc_count:
                    break
        if not made_progress:
            break
        depth += 1

    return _placement(topology, packed_ids), _placement(topology, scattered_ids)


def auto_tpc_counts(topology: Mapping[int, Sequence[int]]) -> tuple[int, ...]:
    """Choose GPC-sized steps plus explicit 40% and 50% probe points."""
    total_tpcs = sum(len(tpcs) for tpcs in topology.values())
    if total_tpcs < 2:
        raise ValueError("at least two TPCs are required for a placement comparison")
    gpc_capacity = max(len(tpcs) for tpcs in topology.values())
    counts = set(range(gpc_capacity, total_tpcs // 2 + 1, gpc_capacity))
    counts.add(math.ceil(total_tpcs * 0.40))
    counts.add(math.ceil(total_tpcs * 0.50))
    return tuple(sorted(count for count in counts if 0 < count < total_tpcs))


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Validate paired SM exposure and summarize packed/scattered bandwidth."""
    by_key: dict[tuple[int, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        key = (int(row["tpc_count"]), int(row["repetition"]))
        placement = str(row["placement"])
        if placement not in ("packed", "scattered"):
            raise ValueError(f"unknown placement {placement!r}")
        by_key.setdefault(key, {})[placement] = row

    paired_by_count: dict[int, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
    for (tpc_count, repetition), pair in sorted(by_key.items()):
        if set(pair) != {"packed", "scattered"}:
            raise ValueError(
                f"TPC count {tpc_count}, repetition {repetition}: "
                "missing packed/scattered pair"
            )
        packed = pair["packed"]
        scattered = pair["scattered"]
        packed_sms = int(packed["observed_sm_count"])
        scattered_sms = int(scattered["observed_sm_count"])
        if packed_sms <= 0 or scattered_sms <= 0:
            raise ValueError(
                f"TPC count {tpc_count}, repetition {repetition}: observed no active SMs"
            )
        if packed_sms != scattered_sms:
            raise ValueError(
                f"TPC count {tpc_count}, repetition {repetition}: "
                f"packed/scattered observed {packed_sms} versus {scattered_sms} SMs"
            )
        if not tpc_count <= packed_sms <= 2 * tpc_count:
            raise ValueError(
                f"TPC count {tpc_count}, repetition {repetition}: observed "
                f"{packed_sms} SMs; expected between {tpc_count} and {2 * tpc_count}"
            )
        paired_by_count.setdefault(tpc_count, []).append((packed, scattered))

    summaries: list[dict] = []
    for tpc_count, pairs in sorted(paired_by_count.items()):
        packed_bw = [float(pair[0]["bandwidth_GBps"]) for pair in pairs]
        scattered_bw = [float(pair[1]["bandwidth_GBps"]) for pair in pairs]
        median_packed = statistics.median(packed_bw)
        median_scattered = statistics.median(scattered_bw)
        summaries.append(
            {
                "tpc_count": tpc_count,
                "paired_repetitions": len(pairs),
                "observed_sm_count": int(pairs[0][0]["observed_sm_count"]),
                "median_packed_GBps": median_packed,
                "median_scattered_GBps": median_scattered,
                "scattered_over_packed": median_scattered / median_packed,
                "scattered_gain_pct": (median_scattered / median_packed - 1.0) * 100.0,
                "scattered_wins": sum(
                    float(scattered["bandwidth_GBps"]) > float(packed["bandwidth_GBps"])
                    for packed, scattered in pairs
                ),
            }
        )
    return summaries


def build_command(
    *,
    nvtaskset: Path,
    benchmark: Path,
    tpcs: Sequence[int],
    buffer_mb: int,
    iterations: int,
    gpu_id: int,
    trials: int,
    load_mode: int,
) -> list[str]:
    if not tpcs:
        raise ValueError("at least one TPC is required")
    raw_mask = 0
    for tpc in tpcs:
        if isinstance(tpc, bool) or not isinstance(tpc, int) or not 0 <= tpc < 128:
            raise ValueError(f"invalid TPC ID {tpc!r} for nvtaskset raw mask")
        raw_mask |= 1 << tpc
    return [
        str(nvtaskset),
        f"0x{raw_mask:x}",
        str(benchmark),
        "--external-single",
        str(buffer_mb),
        str(iterations),
        str(gpu_id),
        str(trials),
        str(load_mode),
    ]


def parse_benchmark_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "external_single":
            return record
    raise ValueError("benchmark output did not contain an external_single JSON record")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare equal-size packed and GPC-scattered TPC placements using "
            "libsmctrl/nvtaskset."
        )
    )
    parser.add_argument(
        "--libsmctrl-dir",
        type=Path,
        required=True,
        help="directory containing nvtaskset, libcuda.so.1, and libsmctrl_test_gpc_info",
    )
    parser.add_argument(
        "--topology-json",
        type=Path,
        help=("previously captured topology JSON; avoids querying nvdebug while "
              "running the MPS benchmark"),
    )
    parser.add_argument(
        "--capture-topology-json",
        type=Path,
        help=("query physical GPC topology, write reusable JSON, and exit before "
              "starting nvtaskset/MPS"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("./green_ctx_bw_bench"),
        help="path to the compiled benchmark",
    )
    parser.add_argument(
        "--tpc-counts",
        default="auto",
        help="comma-separated active-TPC counts, or 'auto' (default)",
    )
    parser.add_argument("--buffer-mb", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--load-mode", type=int, default=1, choices=range(0, 6))
    parser.add_argument("--seed", type=int, default=8228)
    parser.add_argument("--output-prefix", type=Path, default=Path("b200_gpc_bw"))
    return parser


def _prefixed_output_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV file {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_tpc_counts(value: str, topology: Mapping[int, Sequence[int]]) -> tuple[int, ...]:
    if value == "auto":
        return auto_tpc_counts(topology)
    try:
        counts = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    except ValueError as error:
        raise ValueError("--tpc-counts must be 'auto' or comma-separated integers") from error
    if not counts:
        raise ValueError("--tpc-counts did not contain any counts")
    return counts


def _subprocess_environment(libsmctrl_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    old_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{libsmctrl_dir}:{old_library_path}" if old_library_path else str(libsmctrl_dir)
    )
    return environment


def _run_checked(command: Sequence[str], environment: Mapping[str, str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        env=dict(environment),
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(f"command failed with exit {result.returncode}: {rendered}")
    return result


def _query_topology(
    libsmctrl_dir: Path, gpu_id: int, environment: Mapping[str, str]
) -> tuple[dict[int, list[int]], str]:
    tool = libsmctrl_dir / "libsmctrl_test_gpc_info"
    if not tool.is_file():
        raise FileNotFoundError(f"topology tool not found: {tool}")
    result = _run_checked([str(tool), str(gpu_id)], environment)
    return parse_gpc_info(result.stdout), result.stdout


def _run_measurement(
    *,
    args: argparse.Namespace,
    nvtaskset: Path,
    benchmark: Path,
    environment: Mapping[str, str],
    placement_name: str,
    placement: Placement,
    tpc_count: int,
    repetition: int,
    run_order: int,
) -> dict:
    command = build_command(
        nvtaskset=nvtaskset,
        benchmark=benchmark,
        tpcs=placement.tpcs,
        buffer_mb=args.buffer_mb,
        iterations=args.iterations,
        gpu_id=args.gpu_id,
        trials=args.trials,
        load_mode=args.load_mode,
    )
    print(
        f"TPCs={tpc_count} repetition={repetition + 1}/{args.repetitions} "
        f"placement={placement_name} GPCs={len(placement.gpcs)}",
        file=sys.stderr,
    )
    result = _run_checked(command, environment)
    measurement = parse_benchmark_json(result.stdout)
    return {
        "tpc_count": tpc_count,
        "placement": placement_name,
        "gpc_count": len(placement.gpcs),
        "gpc_list": ",".join(str(gpc) for gpc in placement.gpcs),
        "tpc_list": ",".join(str(tpc) for tpc in placement.tpcs),
        "repetition": repetition,
        "run_order": run_order,
        "bandwidth_GBps": float(measurement["bandwidth_GBps"]),
        "observed_sm_count": int(measurement["observed_sm_count"]),
        "observed_sm_ids": ",".join(
            str(sm_id) for sm_id in measurement.get("observed_sm_ids", [])
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    for name in ("buffer_mb", "iterations", "trials", "repetitions"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    libsmctrl_dir = args.libsmctrl_dir.resolve()
    environment = _subprocess_environment(libsmctrl_dir)
    if args.topology_json is not None:
        topology_path = args.topology_json.resolve()
        topology, topology_output = load_topology_json(topology_path)
        topology_source = str(topology_path)
    else:
        topology, topology_output = _query_topology(
            libsmctrl_dir, args.gpu_id, environment
        )
        topology_source = "libsmctrl_test_gpc_info"

    if args.capture_topology_json is not None:
        capture_path = args.capture_topology_json.resolve()
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text(
            json.dumps(
                {
                    "topology": {str(gpc): tpcs for gpc, tpcs in topology.items()},
                    "raw_libsmctrl_output": topology_output,
                    "topology_source": topology_source,
                    "gpu_id": args.gpu_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Captured topology: {capture_path}", file=sys.stderr)
        return 0

    benchmark = args.benchmark.resolve()
    nvtaskset = libsmctrl_dir / "nvtaskset"
    if not nvtaskset.is_file():
        raise FileNotFoundError(f"nvtaskset not found: {nvtaskset}")
    if not benchmark.is_file():
        raise FileNotFoundError(f"benchmark not found: {benchmark}")

    tpc_counts = _parse_tpc_counts(args.tpc_counts, topology)
    print(
        f"Discovered {len(topology)} GPCs and "
        f"{sum(len(tpcs) for tpcs in topology.values())} TPCs; "
        f"testing counts {','.join(map(str, tpc_counts))}",
        file=sys.stderr,
    )

    rng = random.Random(args.seed)
    rows: list[dict] = []
    run_order = 0
    for tpc_count in tpc_counts:
        packed, scattered = build_placements(topology, tpc_count)
        if packed.tpcs == scattered.tpcs:
            raise ValueError(
                f"TPC count {tpc_count} produces identical packed and scattered masks"
            )
        for repetition in range(args.repetitions):
            pair = [("packed", packed), ("scattered", scattered)]
            rng.shuffle(pair)
            for placement_name, placement in pair:
                rows.append(
                    _run_measurement(
                        args=args,
                        nvtaskset=nvtaskset,
                        benchmark=benchmark,
                        environment=environment,
                        placement_name=placement_name,
                        placement=placement,
                        tpc_count=tpc_count,
                        repetition=repetition,
                        run_order=run_order,
                    )
                )
                run_order += 1

    summaries = summarize_rows(rows)
    raw_path = _prefixed_output_path(args.output_prefix, "_raw.csv")
    summary_path = _prefixed_output_path(args.output_prefix, "_summary.csv")
    metadata_path = _prefixed_output_path(args.output_prefix, "_topology.json")
    _write_csv(raw_path, rows)
    _write_csv(summary_path, summaries)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "topology": {str(gpc): tpcs for gpc, tpcs in topology.items()},
                "raw_libsmctrl_output": topology_output,
                "topology_source": topology_source,
                "seed": args.seed,
                "load_mode": args.load_mode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"Raw results: {raw_path}", file=sys.stderr)
    print(f"Summary:     {summary_path}", file=sys.stderr)
    print(f"Topology:    {metadata_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
