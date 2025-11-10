#!/usr/bin/env python3
"""
Simple ordered runner for BlackBox utils scripts.

Usage:
    python run_attack_sequence.py --version 0.5
    python run_attack_sequence.py --version kl --ratio 0.5

Behavior:
- Runs the exact list of scripts in /opt/data/private/BlackBox/utils/ in the specified order.
- Sets environment variables PATCH_VERSION and PATCH_RATIO for each child process.
- Stops on first error (non-zero exit).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# -----------------------
# Configuration (edit if you move files)
# -----------------------
UTILS_DIR = Path("/opt/data/private/BlackBox/utils").resolve()

# Exact ordered list you provided (must match filenames in utils/).
SCRIPT_ORDER = [
    "2_past_patch.py",
    "3-create-data.py",
    # "4-1-detection-clean.py",
    "4-2-detection-patch.py",
    "5-1-json-clean-deformable.py",
    # "5-1-json-clean-detr.py",
    # "5-1-json-clean-dnanchor.py",
    # "5-1-json-clean-sparse.py",
    "5-2-json-patch-deformable.py",
    "5-2-json-patch-detr.py",
    "5-2-json-patch-dnanchor.py",
    "5-2-json-patch-sparse.py",
    "6-filter-clean.py",
    "6-filter-patch.py",
    "7-1-visual-recall.py",
    "7-2-summary.py",
]

PYTHON = sys.executable  # use same python interpreter

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run ordered BlackBox utils scripts with controlled VERSION.")
    p.add_argument("--version", "-v", required=True, help="PATCH_VERSION to pass to child scripts (e.g. '0.5' or 'kl').")
    p.add_argument("--ratio", "-r", default=None, help="Optional PATCH_RATIO to pass to child scripts (e.g. '0.5').")
    p.add_argument("--stop-on-error", dest="stop_on_error", action="store_true", default=True,
                   help="Stop on first script error (default).")
    p.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false",
                   help="Continue even if a script returns non-zero (not recommended).")
    return p.parse_args()

# -----------------------
# Helpers
# -----------------------
def check_utils_dir(path: Path):
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"utils dir not found: {path}")

def build_script_paths(utils_dir: Path, names):
    paths = []
    for nm in names:
        p = utils_dir / nm
        if not p.exists():
            raise FileNotFoundError(f"Expected script not found: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Expected a file but not a regular file: {p}")
        paths.append(p)
    return paths

def run_script(path: Path, env: dict):
    cmd = [PYTHON, str(path)]
    print("\n" + "="*70)
    print(f"Running: {cmd}")
    print(f"cwd: {path.parent}")
    # run with subprocess.run so stdout/stderr stream to console
    res = subprocess.run(cmd, cwd=str(path.parent), env=env)
    print(f"Exit code: {res.returncode}")
    return res.returncode

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()

    try:
        check_utils_dir(UTILS_DIR)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # resolve script paths and fail early if any missing
    try:
        scripts = build_script_paths(UTILS_DIR, SCRIPT_ORDER)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # build environment for child processes
    base_env = os.environ.copy()
    base_env["PATCH_VERSION"] = str(args.version)
    if args.ratio is not None:
        base_env["PATCH_RATIO"] = str(args.ratio)

    print(f"Will run {len(scripts)} scripts in order from: {UTILS_DIR}")
    print("PATCH_VERSION =", base_env.get("PATCH_VERSION"))
    if "PATCH_RATIO" in base_env:
        print("PATCH_RATIO =", base_env.get("PATCH_RATIO"))

    # switch cwd to utils dir for consistent relative paths in child scripts
    os.chdir(str(UTILS_DIR))
    print(f"Changed working dir to {UTILS_DIR}")

    # execute scripts in order
    for script_path in scripts:
        code = run_script(script_path, env=base_env)
        if code != 0:
            print(f"\nERROR: script {script_path.name} exited with code {code}", file=sys.stderr)
            if args.stop_on_error:
                print("Stopping execution due to error.")
                sys.exit(code)
            else:
                print("Continuing despite error (per --no-stop-on-error).")

    print("\nAll scripts completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
