#!/usr/bin/env python3
"""Build, validate, and measure H200/B200 bandwidth in a Runpod CUDA development pod."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GPU_TARGETS = {
    "H200": {"compute_cap": "9.0", "cuda_arch": "sm_90", "minimum_cuda": (12, 4)},
    "B200": {"compute_cap": "10.0", "cuda_arch": "sm_100", "minimum_cuda": (12, 8)},
}


def integer_list(value):
    try:
        return tuple(dict.fromkeys(int(part) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-id", default="0", help="nvidia-smi device index or UUID; CUDA is pinned by UUID")
    p.add_argument("--buffer-mb", type=int, default=2048, help="input size in MiB (default 2048)")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--trials", type=int, default=7)
    p.add_argument("--modes", type=integer_list, default=(1, 4))
    p.add_argument("--blocks-per-sm", type=integer_list, default=(8, 32))
    p.add_argument("--tma-stages", type=integer_list, default=(2, 4, 8))
    p.add_argument("--tma-copy-only", action="store_true", help="TMA transfer/wait control without the SIMT consumer")
    p.add_argument("--tc-n", type=integer_list, default=(8,), help="mode 6 N values: 8,16,32 (4,8,16 FLOPs/input byte)")
    p.add_argument("--tc-consumers", default="mma", help="mode 6 consumers, comma-separated: mma,sum,copy")
    p.add_argument("--tc-schedule", choices=("serial", "overlap", "refill"), default="serial",
                   help="mode 6 MMA schedule; copy/sum keep their control loops")
    p.add_argument("--seed", type=int, default=8228)
    p.add_argument("--sanitize", action="store_true", help="also run memcheck, racecheck, and synccheck")
    p.add_argument("--profile", action="store_true", help="collect Nsight DRAM counters at 8 SMs, threshold, and full GPU")
    p.add_argument("--preflight-only", action="store_true", help="check GPU/toolkit, print JSON, and exit")
    p.add_argument("--output-dir", type=Path, help="new directory for all artifacts")
    return p


def preflight(args):
    for tool in ("nvcc", "nvidia-smi", "make"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is missing; use a CUDA development image with nvcc "
                               "(H200: CUDA 12.4+, B200: CUDA 12.8+)")
    nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
    version = re.search(r"release (\d+)\.(\d+)", nvcc)
    if not version:
        raise RuntimeError("Could not determine the CUDA toolkit version from nvcc --version")
    query = subprocess.check_output([
        "nvidia-smi", "-i", args.gpu_id,
        "--query-gpu=index,name,uuid,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = list(csv.reader(query.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError("Select exactly one H200, B200, or GB200 GPU")
    gpu = dict(zip(("index", "name", "uuid", "compute_cap", "driver_version", "memory_mib"),
                   (field.strip() for field in rows[0])))
    model = re.search(r"\b(H200|B200|GB200)\b", gpu["name"].upper())
    if not model:
        raise RuntimeError(f"Expected an H200, B200, or GB200, got {gpu['name']} ({gpu['compute_cap']})")
    family = "B200" if model.group(1) == "GB200" else model.group(1)
    target = GPU_TARGETS[family]
    if 6 in args.modes and family != "B200":
        raise RuntimeError("mode 6 requires B200/GB200 and a sm_100a build")
    if gpu["compute_cap"] != target["compute_cap"]:
        raise RuntimeError(f"Expected {family} compute capability {target['compute_cap']}, "
                           f"got {gpu['compute_cap']}")
    if tuple(map(int, version.groups())) < target["minimum_cuda"]:
        minimum = ".".join(map(str, target["minimum_cuda"]))
        raise RuntimeError(f"{family} requires an nvcc toolkit of CUDA {minimum} or newer")
    tools = {}
    for requested, tool in ((args.sanitize, "compute-sanitizer"), (args.profile, "ncu")):
        if requested:
            if not shutil.which(tool):
                raise RuntimeError(f"{tool} is required for the requested validation; install matching CUDA toolkit tools")
            tools[tool] = {"path": shutil.which(tool), "version": subprocess.check_output(
                [tool, "--version"], text=True, stderr=subprocess.STDOUT)}
    return {"gpu": gpu, "gpu_family": family,
            "cuda_arch": "sm_100a" if 6 in args.modes else target["cuda_arch"], "nvcc_version": nvcc,
            "nvcc_path": shutil.which("nvcc"), "validation_tools": tools}


def configurations(args):
    for mode in args.modes:
        for blocks in args.blocks_per_sm:
            for stages in (args.tma_stages if mode in (4, 6) else (2,)):
                for n in (args.tc_n if mode == 6 else (0,)):
                    for consumer in (args.tc_consumers if mode == 6 else ("none",)):
                        yield mode, blocks, stages, n, consumer


def consumer_options(args, mode, n, consumer):
    if mode == 6:
        return ["--tc-n", n, "--tc-consumer", consumer, "--tc-schedule", args.tc_schedule]
    return ["--tma-copy-only"] if mode == 4 and args.tma_copy_only else []


def parse_ncu_report(path, input_bytes):
    """Validate one profiled kernel and normalize the raw CSV's explicit units."""
    values, kernels = {}, set()
    header = None
    with path.open() as handle:
        for row in csv.reader(handle):
            if "Metric Name" in row and "Metric Value" in row:
                header = row
                continue
            if header is None or len(row) != len(header):
                continue
            record = dict(zip(header, row))
            if "read_bandwidth_kernel" not in record.get("Kernel Name", ""):
                continue
            kernels.add((record.get("ID"), record["Kernel Name"]))
            values[record["Metric Name"]] = (record["Metric Value"], record["Metric Unit"])
    if len(kernels) != 1:
        raise RuntimeError(f"Expected counters for one bandwidth kernel in {path}")
    scales = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
              "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1, "%": 1}

    def counter(name):
        try:
            number, unit = values[name]
            value = float(number.replace(",", "")) * scales[unit]
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"Missing or invalid counter {name} in {path}") from error
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Invalid counter {name} in {path}")
        return value

    read = counter("dram__bytes_read.sum")
    write = counter("dram__bytes_write.sum")
    seconds = counter("gpu__time_duration.sum")
    throughput = counter("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    if seconds <= 0:
        raise RuntimeError(f"Invalid duration counter in {path}")
    return {"report": str(path), "dram_read_bytes": read, "dram_write_bytes": write,
            "kernel_seconds": seconds, "dram_read_GBps": read / seconds / 1e9,
            "dram_throughput_pct": throughput, "dram_read_fraction_of_input": read / input_bytes}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.buffer_mb, args.iterations, args.trials) <= 0 or not 0 <= args.seed <= 2147483647:
        p.error("buffer size, iterations, and trials must be positive; seed must be in 0..2147483647")
    if any(m not in range(7) for m in args.modes):
        p.error("--modes must contain values from 0 through 6")
    if any(not 1 <= b <= 256 for b in args.blocks_per_sm):
        p.error("--blocks-per-sm must contain values in 1..256")
    if any(s not in (2, 4, 8) for s in args.tma_stages):
        p.error("--tma-stages must contain only 2, 4, or 8")
    if any(n not in (8, 16, 32) for n in args.tc_n):
        p.error("--tc-n must contain only 8, 16, or 32")
    args.tc_consumers = tuple(dict.fromkeys(args.tc_consumers.split(",")))
    if any(c not in ("mma", "sum", "copy") for c in args.tc_consumers):
        p.error("--tc-consumers must contain only mma, sum, or copy")
    meta = preflight(args)
    if args.preflight_only:
        print(json.dumps(meta, indent=2))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.output_dir or ROOT / f"{meta['gpu_family'].lower()}_results" / timestamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "manifest.json"
    meta.update({"started_utc": timestamp, "complete": False, "runs": [],
                 "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=meta["gpu"]["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    meta["environment"] = {k: v for k, v in environment.items() if
                           k.startswith("CUDA_") or k == "NVIDIA_VISIBLE_DEVICES"}

    def save():
        manifest.write_text(json.dumps(meta, indent=2) + "\n")

    def run(command, name, timeout=1800, stdout_path=None, working_dir=None):
        record = {"command": list(map(str, command)), "name": name}
        meta["runs"].append(record)
        save()
        print(shlex.join(record["command"]), flush=True)
        stdout_path = stdout_path or out / f"{name}.stdout.log"
        with stdout_path.open("w") as stdout, (out / f"{name}.stderr.log").open("w") as stderr:
            result = subprocess.run(record["command"], cwd=working_dir or ROOT, env=environment,
                                    stdout=stdout, stderr=stderr, timeout=timeout)
        record["exit_code"] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {out / (name + '.stderr.log')}")
        return stdout_path

    try:
        source = out / "source"
        source.mkdir()
        meta["source_sha256"] = {}
        for name in ("green_ctx_bw_bench.cu", "bench_options.h", "tma_tensor_core.cuh", "Makefile",
                     "run_b200_runpod.py", "run_h200_runpod.py", "plot_results.py"):
            shutil.copy2(ROOT / name, source / name)
            meta["source_sha256"][name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        binary = out / "green_ctx_bw_bench"
        run(["make", "-B", "all", f"CUDA_ARCH={meta['cuda_arch']}", f"NVCC={meta['nvcc_path']}", f"TARGET={binary}"],
            "build", working_dir=source)
        probe_file = run([binary, "--external-probe", "0"], "residency")
        meta["residency_probe"] = json.loads(probe_file.read_text())
        total_sms = meta["residency_probe"]["total_sm_count"]
        validation_args = argparse.Namespace(**vars(args))
        validation_args.modes = tuple(range(7 if 6 in args.modes else 6))
        validation_args.blocks_per_sm = (1,)
        for mode, _, stages, n, consumer in configurations(validation_args):
            extra = consumer_options(args, mode, n, consumer)
            suffix = f"_n{n}_{consumer}_{args.tc_schedule}" if mode == 6 else ""
            command = [binary, "--validate-only", *extra, "--tma-stages", stages, "1", "1", "0", "0", "1", mode]
            name = f"validate_mode{mode}_stages{stages}{suffix}"
            run(command, name, timeout=180)
            if args.sanitize:
                for tool in ("memcheck", "racecheck", "synccheck"):
                    run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "1", *command],
                        f"{name}_{tool}", timeout=600)
        for mode, blocks, stages, n, consumer in configurations(args):
            suffix = f"_n{n}_{consumer}_{args.tc_schedule}" if mode == 6 else ""
            name = f"mode{mode}_blocks{blocks}_stages{stages}{suffix}"
            tuning = ["--blocks-per-sm", blocks, "--tma-stages", stages, "--seed", args.seed]
            tuning += consumer_options(args, mode, n, consumer)
            run([binary, "--require-dram", *tuning, args.buffer_mb, args.iterations,
                 "0", "0", args.trials, mode], name, stdout_path=out / f"{name}.csv")
            run([sys.executable, source / "plot_results.py", out / f"{name}.csv", out / f"{name}.png"], f"{name}_plot")
            if args.profile:
                with (out / f"{name}.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                threshold = 0.95 * float(rows[0]["bandwidth_GBps"])
                saturation = min(int(r["sm_count_allocated"]) for r in rows if float(r["bandwidth_GBps"]) >= threshold)
                for sms in sorted({8, saturation, total_sms}):
                    profile_name = f"{name}_sm{sms}_profile"
                    run(["ncu", "--kernel-name", "regex:read_bandwidth_kernel", "--launch-skip", "5",
                         "--launch-count", "1", "--cache-control", "none", "--clock-control", "none",
                         "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed",
                         "--csv", "--page", "raw", "--units", "base", "--log-file", out / f"{profile_name}.csv",
                         binary, "--skip-validation", "--require-dram", "--sm-count", sms, *tuning,
                         args.buffer_mb, "1", "0", "0", "1", mode], profile_name)
                    counters = parse_ncu_report(out / f"{profile_name}.csv", args.buffer_mb * 1024**2)
                    meta["runs"][-1]["checked_counters"] = counters
                    save()
                    print(f"{profile_name}: DRAM/input bytes={counters['dram_read_fraction_of_input']:.3f}, "
                          f"DRAM read={counters['dram_read_GBps']:.2f} GB/s", flush=True)
        meta["complete"] = True
    except Exception as error:
        meta["error"] = str(error)
        raise
    finally:
        save()
    print(f"Results saved to {out}")
    return 0


def cli():
    try:
        return main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
