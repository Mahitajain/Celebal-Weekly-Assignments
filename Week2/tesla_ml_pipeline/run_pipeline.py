#!/usr/bin/env python3
"""
run_pipeline.py
───────────────────────────────────────────────────────────────
🚀  Single entry point – runs the full Tesla EV ML pipeline.

Usage
─────
    python run_pipeline.py              # run all stages
    python run_pipeline.py --stage 1    # run only stage 1
    python run_pipeline.py --stage 1 4  # run stages 1 and 4
    python run_pipeline.py --skip 6     # run all except stage 6

Stages
──────
  1  Preprocessing
  2  EDA
  3  Feature Engineering
  4  Regression Modeling
  5  Hyperparameter Tuning
  6  Time Series Forecasting
  7  Report Generation
"""

import sys, os, time, argparse, traceback

# Ensure src/ is on the path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

import src.utils as _u   # warm-up paths

STAGES = {
    1: ("Preprocessing",          "src.01_preprocessing"),
    2: ("EDA",                    "src.02_eda"),
    3: ("Feature Engineering",    "src.03_feature_engineering"),
    4: ("Regression Modeling",    "src.04_regression_modeling"),
    5: ("Hyperparameter Tuning",  "src.05_hyperparameter_tuning"),
    6: ("Time Series Forecast",   "src.06_time_series_forecasting"),
    7: ("Report Generation",      "src.generate_report"),
}


def run_stage(stage_num: int) -> bool:
    name, module_path = STAGES[stage_num]
    print(f"\n{'━' * 65}")
    print(f"  STAGE {stage_num} / {max(STAGES)} – {name}")
    print(f"{'━' * 65}")
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_path)
        mod.run()
        elapsed = time.time() - t0
        print(f"\n  ✅  Stage {stage_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌  Stage {stage_num} FAILED after {elapsed:.1f}s")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Tesla EV ML Pipeline")
    parser.add_argument("--stage", nargs="+", type=int,
                        help="Run only these stage numbers (e.g. --stage 1 3 4)")
    parser.add_argument("--skip",  nargs="+", type=int,
                        help="Skip these stage numbers (e.g. --skip 5 6)")
    args = parser.parse_args()

    if args.stage:
        to_run = sorted(args.stage)
    else:
        to_run = list(STAGES.keys())

    if args.skip:
        to_run = [s for s in to_run if s not in args.skip]

    print("\n" + "═" * 65)
    print("  🚗  TESLA EV ML PIPELINE  –  FULL RUN")
    print("═" * 65)
    print(f"  Stages to run: {to_run}")
    print(f"  Outputs dir  : {_u.OUTPUTS_DIR}")
    print(f"  Models dir   : {_u.MODELS_DIR}")
    print(f"  Reports dir  : {_u.REPORTS_DIR}")

    total_start = time.time()
    results = {}

    for s in to_run:
        if s not in STAGES:
            print(f"  ⚠  Unknown stage {s} – skipping")
            continue
        results[s] = run_stage(s)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - total_start
    print("\n" + "═" * 65)
    print("  PIPELINE SUMMARY")
    print("═" * 65)
    for s, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status}  Stage {s}: {STAGES[s][0]}")
    print(f"\n  Total elapsed: {total:.1f}s")

    n_failed = sum(1 for ok in results.values() if not ok)
    if n_failed == 0:
        print("\n  🎉  All stages completed successfully!")
        print(f"  📊  Plots  → {_u.OUTPUTS_DIR}")
        print(f"  🤖  Models → {_u.MODELS_DIR}")
        print(f"  📝  Report → {os.path.join(_u.REPORTS_DIR, 'pipeline_report.md')}")
    else:
        print(f"\n  ⚠  {n_failed} stage(s) failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
