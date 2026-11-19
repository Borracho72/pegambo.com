"""Generate all realizations/analyses and consolidate results.json."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--reuse-fields", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="run a small end-to-end check; not a statistical reproduction")
    p.add_argument("--output-dir", type=Path,
                   help="default: reproduced/ for full runs or smoke_output/ for smoke runs")
    opts = p.parse_args()
    root = Path(__file__).resolve().parent
    out = (opts.output_dir or Path("smoke_output" if opts.smoke else "reproduced")).resolve()
    if out == root:
        p.error("use an output directory other than the package directory")
    out.mkdir(parents=True, exist_ok=True)
    logs = out/"logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
           "PYTHONDONTWRITEBYTECODE": "1"}

    def run(log, script, *args):
        print(f"{script} → {logs.name}/{log}", flush=True)
        with (logs/log).open("w") as fd:
            subprocess.run([sys.executable, root/script, *args], cwd=out, env=env,
                           stdout=fd, stderr=subprocess.STDOUT, check=True)

    main_fields, fine_fields = (8, 8) if opts.smoke else (64, 32)
    sample_args = ("--width", "16") if opts.smoke else ()
    analysis_args = ("--reps", "48") if opts.smoke else ()
    if not opts.assemble_only:
        if not opts.reuse_fields:
            run("simulation.log", "simulate.py", "--fields", str(main_fields),
                "--dir", "fields", *sample_args)
            run("simulation_high.log", "simulate.py", "--fields", str(fine_fields),
                "--fine", ".0125", "--dir", "fields_high", *sample_args)
        run("analyze.log", "analyze.py", "--dir", "fields", "--limit", str(main_fields),
            "--out", "results_base.json", *analysis_args)
        run("analyze_verified.log", "analyze.py", "--dir", "fields",
            "--limit", str(main_fields), "--verified-only", "--skip-overlay",
            "--out", "results_verified.json", *analysis_args)
        run("analyze_high.log", "analyze.py", "--dir", "fields_high",
            "--limit", str(fine_fields), "--verified-only", "--skip-overlay",
            "--out", "results_high.json", *analysis_args)
    result = json.loads((out/"results_base.json").read_text())
    result["checks"] = {}
    for name in ("verified", "high"):
        part = json.loads((out/f"results_{name}.json").read_text())
        result["checks"][name] = {k:v for k,v in part.items() if k in ("simulation","simulations")}
    if opts.smoke:
        result["smoke_test"] = True
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Complete: {out.name}/results.json", flush=True)


if __name__ == "__main__":
    main()
