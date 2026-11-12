#!/usr/bin/env python
import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser(description="Run tau sweep for EMT vs Beyond-EMT scripts.")
    ap.add_argument("--taus", type=str, default="0.01,0.03,0.05,0.10",
                    help="Comma-separated list of outbreak thresholds tau.")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use.")
    args = ap.parse_args()

    scripts = [
        "compare_EMT_tau_sweep.py",
        "compare_mixing_tau_sweep.py",
        "aggregate_para_tau_sweep.py",
    ]

    for s in scripts:
        print("\n==============================")
        print("Running:", s)
        cmd = [args.python, s, "--taus", args.taus]
        print("CMD:", " ".join(cmd))
        subprocess.check_call(cmd)

    print("\nDONE. Check output folders:")
    print(" - beyond_emt_results/")
    print(" - beyond_emt_results_param_vs_mixing/")
    print(" - agg_param_forecast_out/")

if __name__ == "__main__":
    main()
