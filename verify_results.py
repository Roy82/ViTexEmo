# verify_results.py
# This script compares the reproduced outputs with the expected results
# to confirm the reproducibility of the ViTexEmo framework.

import os
import argparse
import pandas as pd

def compare_csv(expected_path, observed_path, tol=0.5):
    """Compare two CSV files and return True if they match within tolerance."""
    if not os.path.exists(expected_path):
        print(f"Expected file missing: {expected_path}")
        return False

    if not os.path.exists(observed_path):
        print(f"Observed file missing: {observed_path}")
        return False

    exp = pd.read_csv(expected_path)
    obs = pd.read_csv(observed_path)

    # Ensure same columns
    if list(exp.columns) != list(obs.columns):
        print(f"Column mismatch in {os.path.basename(observed_path)}")
        return False

    # Check numeric differences
    for col in exp.columns:
        diffs = (exp[col] - obs[col]).abs()
        if (diffs > tol).any():
            print(f"Mismatch in column '{col}' of {os.path.basename(observed_path)}.")
            print(f"Max difference: {diffs.max()}")
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify reproducibility results.")
    parser.add_argument("--expected", required=True, help="Folder with expected CSV results")
    parser.add_argument("--observed", required=True, help="Folder with reproduced output CSV results")
    parser.add_argument("--tol", type=float, default=0.5, help="Allowed numerical tolerance")
    args = parser.parse_args()

    checks = [
        ("textual_metrics_table15.csv", "textual_metrics.csv"),
        ("visual_metrics_table16.csv", "visual_metrics.csv"),
        ("overall_metrics.csv", "overall_metrics.csv"),
    ]

    all_passed = True

    for expected_file, observed_file in checks:
        expected_path = os.path.join(args.expected, expected_file)
        observed_path = os.path.join(args.observed, observed_file)

        print(f"Comparing {expected_file} vs {observed_file} ...")

        if not compare_csv(expected_path, observed_path, tol=args.tol):
            all_passed = False
            break

    if all_passed:
        print("============================================")
        print("          REPRODUCTION PASSED ✔️")
        print("============================================")
    else:
        print("============================================")
        print("          REPRODUCTION FAILED ❌")
        print("============================================")
        exit(1)
