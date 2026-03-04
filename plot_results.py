#!/usr/bin/env python3
"""
plot_results.py — Visualize Green Context BW saturation benchmark results.

Reads CSV output from green_ctx_bw_bench and produces:
  1. Bandwidth vs SM count plot
  2. Annotated saturation point (95% of peak)

Usage:
    python3 plot_results.py results.csv [output.png]
"""

import sys
import csv
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <results.csv> [output.png]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "bw_saturation.png"

    # --- Read CSV ---
    sm_requested = []
    sm_allocated = []
    bandwidth = []
    pct_of_peak = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sm_requested.append(int(row["sm_count_requested"]))
            sm_allocated.append(int(row["sm_count_allocated"]))
            bandwidth.append(float(row["bandwidth_GBps"]))
            pct_of_peak.append(float(row["pct_of_full_gpu"]))

    if not bandwidth:
        print("ERROR: No data in CSV", file=sys.stderr)
        sys.exit(1)

    peak_bw = max(bandwidth)
    full_gpu_bw = bandwidth[0]  # first row is always full-GPU

    # --- Find saturation point (first SM count >= 95% of full-GPU BW) ---
    threshold = 0.95 * full_gpu_bw
    sat_idx = None
    # Skip index 0 (full GPU), look through sorted-by-SM data
    sorted_data = sorted(zip(sm_allocated, bandwidth, sm_requested),
                         key=lambda x: x[0])

    for i, (sms, bw, req) in enumerate(sorted_data):
        if bw >= threshold:
            sat_idx = i
            break

    sat_sms = sorted_data[sat_idx][0] if sat_idx is not None else None
    sat_bw = sorted_data[sat_idx][1] if sat_idx is not None else None

    # --- Print text summary ---
    print(f"\n{'='*60}")
    print(f"  Green Context BW Saturation Analysis")
    print(f"{'='*60}")
    print(f"  Full-GPU bandwidth:     {full_gpu_bw:.2f} GB/s")
    print(f"  Peak measured BW:       {peak_bw:.2f} GB/s")
    print(f"  95% threshold:          {threshold:.2f} GB/s")
    if sat_sms is not None:
        total_sms = max(sm_allocated)
        print(f"  Saturation point:       {sat_sms} SMs "
              f"({sat_sms/total_sms*100:.1f}% of {total_sms} total)")
        print(f"  BW at saturation:       {sat_bw:.2f} GB/s "
              f"({sat_bw/full_gpu_bw*100:.1f}% of full-GPU)")
    else:
        print(f"  Saturation point:       NOT REACHED")
    print(f"{'='*60}\n")

    # --- Print table ---
    print(f"{'SM(req)':>8} {'SM(alloc)':>10} {'BW(GB/s)':>10} {'%peak':>8}")
    print(f"{'-'*8:>8} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8}")
    for req, alloc, bw, pct in zip(sm_requested, sm_allocated, bandwidth, pct_of_peak):
        marker = " <-- 95%" if sat_sms is not None and alloc == sat_sms and bw >= threshold else ""
        print(f"{req:>8} {alloc:>10} {bw:>10.2f} {pct:>7.1f}%{marker}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nWARNING: matplotlib not available, skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))

    # Sort by allocated SMs for plotting
    plot_sms = [d[0] for d in sorted_data]
    plot_bw = [d[1] for d in sorted_data]

    ax1.plot(plot_sms, plot_bw, "o-", color="#2196F3", linewidth=2,
             markersize=6, label="Measured Read BW")

    # Threshold line
    ax1.axhline(y=threshold, color="#FF9800", linestyle="--", linewidth=1.5,
                label=f"95% of full-GPU ({threshold:.0f} GB/s)")

    # Full-GPU line
    ax1.axhline(y=full_gpu_bw, color="#4CAF50", linestyle=":", linewidth=1.5,
                label=f"Full-GPU BW ({full_gpu_bw:.0f} GB/s)")

    # Mark saturation point
    if sat_sms is not None:
        ax1.axvline(x=sat_sms, color="#F44336", linestyle="--", alpha=0.7,
                    linewidth=1.5)
        ax1.annotate(f"Saturation: {sat_sms} SMs\n({sat_bw:.0f} GB/s)",
                     xy=(sat_sms, sat_bw),
                     xytext=(sat_sms + max(plot_sms)*0.05, sat_bw * 0.85),
                     fontsize=11, fontweight="bold", color="#F44336",
                     arrowprops=dict(arrowstyle="->", color="#F44336",
                                    linewidth=1.5),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               edgecolor="#F44336", alpha=0.9))

    ax1.set_xlabel("Number of SMs (allocated)", fontsize=13)
    ax1.set_ylabel("Read Bandwidth (GB/s)", fontsize=13)
    ax1.set_title("DRAM Read BW Saturation vs. SM Count\n"
                  "(CUDA Green Contexts)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Secondary axis: percentage of total SMs
    total_sms_val = max(plot_sms)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    tick_positions = [s for s in plot_sms if s % (total_sms_val // 4) < max(1, total_sms_val // 8)]
    if not tick_positions:
        tick_positions = plot_sms[::max(1, len(plot_sms)//6)]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{s/total_sms_val*100:.0f}%" for s in tick_positions])
    ax2.set_xlabel("Percentage of Total SMs", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
